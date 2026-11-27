import * as THREE from "three";
import { STLLoader } from "../../vendor/three/examples/jsm/loaders/STLLoader.js";
import { EXRLoader } from "../../vendor/three/examples/jsm/loaders/EXRLoader.js";

const CONFIG = Object.freeze({
  initialHold: 0.6 / 2.25,
  rotationSeconds: 1,
  holdSeconds: 1.1 / 2.25,
  cameraIdleSeconds: 5,
  cameraReturnSeconds: 0.9,
  azimuth: Math.atan2(-4, 3),
  elevation: Math.atan2(2.7, 5),
  topSpan: 2.2,
  obliqueSpan: 2.4,
  background: 0x141414,
  yellow: 0xffd65a,
  blue: 0x339fe5,
});

const assets = new URL("../../assets/models/", import.meta.url);
const clamp = THREE.MathUtils.clamp;
const ease = (value) => {
  const t = clamp(value, 0, 1);
  return t * t * t * (10 + t * (-15 + 6 * t));
};

// 所有视角共同使用真实时间，每段恰好转过一个对称阶的角度。
export function rotationAt(seconds, fold = 3) {
  const { initialHold, rotationSeconds, holdSeconds } = CONFIG;
  if (seconds <= initialHold) return { angle: 0, reference: 1, current: 1 };

  const elapsed = seconds - initialHold;
  const period = rotationSeconds + holdSeconds;
  const segment = Math.floor(elapsed / period);
  const localTime = elapsed - segment * period;
  const angle = -((segment % fold) + ease(localTime / rotationSeconds)) * 2 * Math.PI / fold;
  return { angle, reference: 1, current: 1 };
}

async function loadModels() {
  const response = await fetch(new URL("rotation_axes.json", assets));
  if (!response.ok) throw new Error(`旋转轴文件读取失败：${response.status}`);
  const { models } = await response.json();
  const loader = new STLLoader();
  return Promise.all([models.trellis2, models.symtrellis].map(async (entry) => {
    const normalsName = entry.source.replace(/\.stl$/i, ".normals.f32");
    const [geometry, normalResponse] = await Promise.all([
      loader.loadAsync(new URL(entry.source, assets).href),
      fetch(new URL(normalsName, assets)),
    ]);
    if (!normalResponse.ok) throw new Error(`法线文件读取失败：${normalResponse.status}`);
    const normals = new Float32Array(await normalResponse.arrayBuffer());
    if (normals.length !== geometry.attributes.position.array.length) {
      throw new Error("平滑法线数量与网格顶点数量不同");
    }
    geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    const point = new THREE.Vector3(...entry.axis_point_at_z0);
    const axis = new THREE.Vector3(...entry.axis_direction).normalize();
    const alignment = new THREE.Quaternion().setFromUnitVectors(axis, new THREE.Vector3(0, 0, 1));

    // 只把检测出的旋转中心移到原点，把轴摆正。不修改网格形状。
    geometry.translate(-point.x, -point.y, -point.z);
    geometry.applyQuaternion(alignment);
    geometry.normalizeNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }));
}

function makeStudioEnvironment(renderer, texture) {
  const generator = new THREE.PMREMGenerator(renderer);
  const environment = generator.fromEquirectangular(texture);
  generator.dispose();
  return environment;
}

function makeFeatureMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { viewSpan: { value: 1 } },
    side: THREE.DoubleSide,
    toneMapped: false,
    vertexShader: `
      varying vec3 viewNormal;
      varying float viewDepth;
      void main() {
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        viewNormal = normalMatrix * normal;
        viewDepth = -viewPosition.z;
        gl_Position = projectionMatrix * viewPosition;
      }
    `,
    fragmentShader: `
      uniform float viewSpan;
      varying vec3 viewNormal;
      varying float viewDepth;

      // 两个通道保存单位法线，第三个通道保存归一化的正交视深。
      vec2 encodeNormal(vec3 n) {
        n /= max(abs(n.x) + abs(n.y) + abs(n.z), 0.0001);
        if (n.z < 0.0) {
          vec2 direction = vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
          n.xy = (1.0 - abs(n.yx)) * direction;
        }
        return n.xy;
      }

      void main() {
        vec3 n = normalize(viewNormal);
        if (!gl_FrontFacing) n = -n;
        gl_FragColor = vec4(encodeNormal(n), (viewDepth - 5.0) / viewSpan, 1.0);
      }
    `,
  });
}

function makePanel(geometry, material, environment, topView, perspective = false, renderOutlines = true) {
  const scene = new THREE.Scene();
  scene.environment = environment;
  // Blender 的环境贴图以 Z 轴为上方，Three.js 的环境贴图以 Y 轴为上方。
  scene.environmentRotation.x = Math.PI / 2;
  scene.environmentIntensity = 1.15;
  const object = new THREE.Mesh(geometry, material);
  scene.add(object);
  const fill = new THREE.DirectionalLight(0xffffff, 0.65);
  fill.position.set(3, -4, 5);
  scene.add(fill, new THREE.HemisphereLight(0xffffff, 0x687180, 0.2));

  const camera = perspective
    ? new THREE.PerspectiveCamera(24, 1, 0.1, 20)
    : new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20);
  if (!renderOutlines) return { scene, object, camera, topView };

  const maskScene = new THREE.Scene();
  const mask = new THREE.Mesh(geometry, makeFeatureMaterial());
  maskScene.add(mask);
  const beauty = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
  const original = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
  const rotated = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
  return { scene, object, maskScene, mask, camera, beauty, original, rotated, topView };
}

function makeCompositor(background, referenceColor, staggered) {
  const uniforms = {
    beauty: { value: null }, original: { value: null }, rotated: { value: null },
    pixel: { value: new THREE.Vector2() },
    referenceOpacity: { value: 0 }, currentOpacity: { value: 0 },
    background: { value: new THREE.Color(background) },
    yellow: { value: new THREE.Color(referenceColor) },
    blue: { value: new THREE.Color(CONFIG.blue) },
  };
  const material = new THREE.ShaderMaterial({
    uniforms,
    defines: staggered ? { STAGGERED_PANELS: 1 } : {},
    depthTest: false,
    depthWrite: false,
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D beauty, original, rotated;
      uniform vec2 pixel;
      uniform float referenceOpacity, currentOpacity;
      uniform vec3 background, yellow, blue;
      varying vec2 vUv;

      vec3 decodeNormal(vec2 encoded) {
        vec3 n = vec3(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
        float correction = clamp(-n.z, 0.0, 1.0);
        n.xy += vec2(n.x >= 0.0 ? -correction : correction,
                     n.y >= 0.0 ? -correction : correction);
        return normalize(n);
      }

      // 每个姿态单独进行深度测试，只提取该相机实际可见的结构边界。
      float featureLines(sampler2D mask) {
        vec2 d = pixel;
        vec4 center = texture2D(mask, vUv);
        float low = center.a;
        float high = low;
        float interior = 0.0;
        vec3 n = decodeNormal(center.rg);
        vec2 directions[4];
        directions[0] = vec2(1.0, 0.0);
        directions[1] = vec2(0.0, 1.0);
        directions[2] = vec2(0.7071, 0.7071);
        directions[3] = vec2(0.7071, -0.7071);
        for (int i = 0; i < 4; i++) {
          vec2 offset = directions[i] * d;
          vec4 a = texture2D(mask, vUv - offset);
          vec4 b = texture2D(mask, vUv + offset);
          low = min(low, min(a.a, b.a));
          high = max(high, max(a.a, b.a));
          if (center.a > 0.97 && a.a > 0.97 && b.a > 0.97) {
            float depthJump = max(abs(center.b - a.b), abs(center.b - b.b));
            float depthBend = abs(a.b + b.b - 2.0 * center.b);
            vec3 na = decodeNormal(a.rg);
            vec3 nb = decodeNormal(b.rg);
            float normalJump = 1.0 - min(dot(n, na), dot(n, nb));
            float normalBend = length(na + nb - 2.0 * n);
            // 排除连续斜面上的深度梯度和光滑曲面上的连续法线变化。
            float occlusion = smoothstep(0.012, 0.032, depthJump) * smoothstep(0.005, 0.016, depthBend);
            float crease = smoothstep(0.055, 0.19, normalJump) * smoothstep(0.09, 0.30, normalBend);
            interior = max(interior, max(occlusion, crease));
          }
        }
        float silhouette = smoothstep(0.08, 0.92, high - low);
        return max(silhouette, interior);
      }

      void main() {
        vec4 surface = texture2D(beauty, vUv);
        vec3 modelColor = surface.rgb / max(surface.a, 0.0001);
        #ifdef TONE_MAPPING
          modelColor = toneMapping(modelColor);
        #endif
        vec3 color = mix(background, modelColor, surface.a);
        float a = referenceOpacity > 0.0 ? referenceOpacity * featureLines(original) : 0.0;
        float b = currentOpacity > 0.0 ? currentOpacity * featureLines(rotated) : 0.0;
        #ifdef STAGGERED_PANELS
          // 相邻视图的空白区域不能盖住另一个视图的网格或轮廓。
          if (max(surface.a, max(a, b)) < 0.001) discard;
        #endif
        // 重合处混合两种颜色，不用后绘制的轮廓覆盖前一条。
        vec3 line = (a * yellow + b * blue) / max(a + b, 0.0001);
        color = mix(color, line, max(a, b));
        gl_FragColor = vec4(color, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));
  return { scene, camera: new THREE.Camera(), uniforms };
}

export async function createSpinningTopScene(canvas, {
  staticComparison = false, outlinePixels = 1.05, rowsByMethod = false,
} = {}) {
  const models = await loadModels();
  const material = new THREE.MeshStandardMaterial({
    color: 0xa79f96, metalness: 0.94, roughness: 0.24, side: THREE.DoubleSide,
  });
  const items = [
    { geometry: models[0], material, topView: true, fold: 3 },
    { geometry: models[1], material, topView: true, fold: 3 },
    { geometry: models[0], material, topView: false, fold: 3 },
    { geometry: models[1], material, topView: false, fold: 3 },
  ];
  return createRotatingComparison(canvas, {
    columns: 2,
    rows: 2,
    outlinePixels,
    fixedAngle: staticComparison ? -2 * Math.PI / 3 : null,
    staggered: staticComparison,
    background: staticComparison ? 0xffffff : CONFIG.background,
    referenceColor: staticComparison ? 0xd6a000 : CONFIG.yellow,
    items: rowsByMethod ? [items[0], items[2], items[1], items[3]] : items,
  });
}

// 第四页与第五页共用完全相同的灯光、轮廓计算和相机交互。
export async function createRotatingComparison(canvas, {
  columns, rows, outlinePixels: outlineWidth, items, animated = true, idleSpinPeriodSeconds = 0,
  renderOutlines = true,
  fixedAngle = null, background = CONFIG.background, referenceColor = CONFIG.yellow,
  staggered = false,
}) {
  const studioTexture = await new EXRLoader().setDataType(THREE.HalfFloatType)
    .loadAsync(new URL("../../assets/images/neutral_studio.exr", import.meta.url).href);
  studioTexture.mapping = THREE.EquirectangularReflectionMapping;
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: !renderOutlines, powerPreference: "high-performance", preserveDrawingBuffer: fixedAngle !== null });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.toneMappingExposure = 1;
  let environment = makeStudioEnvironment(renderer, studioTexture);
  const panels = items.map((item) => ({
    ...makePanel(item.geometry, item.material, environment.texture, item.topView, item.perspective, renderOutlines),
    fold: item.fold,
    fittingRadius: item.fittingRadius,
    cameraDistance: item.cameraDistance ?? 5,
  }));
  const compositor = renderOutlines ? makeCompositor(background, referenceColor, staggered) : null;
  let frame = null;
  let startTime = 0;
  let lastInteraction = -Infinity;
  let azimuth = CONFIG.azimuth;
  let elevation = CONFIG.elevation;
  let returnFrom = null;
  let drag = null;
  let previousView = null;
  let previousRotations = null;
  let idleSpinAngle = 0;
  let lastIdleSpinFrame = null;
  let staticRunning = false;

  canvas.addEventListener("webglcontextrestored", () => {
    environment.dispose();
    environment = makeStudioEnvironment(renderer, studioTexture);
    panels.forEach((panel) => { panel.scene.environment = environment.texture; });
    previousView = null;
    previousRotations = null;
    if (fixedAngle !== null) scheduleStaticRender();
  });

  function updateCamera(panel, aspect) {
    const { camera, topView, cameraDistance } = panel;
    const span = panel.fittingRadius
      ? 2.16 * panel.fittingRadius / Math.min(1, aspect)
      : (topView ? CONFIG.topSpan : CONFIG.obliqueSpan);
    if (camera.isPerspectiveCamera) {
      // 在模型中心平面保持与正交相同的取景范围。
      camera.aspect = aspect;
      camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(span / (2 * cameraDistance)));
    } else {
      camera.left = -span * aspect / 2;
      camera.right = span * aspect / 2;
      camera.top = span / 2;
      camera.bottom = -span / 2;
    }
    camera.updateProjectionMatrix();
    if (panel.mask) panel.mask.material.uniforms.viewSpan.value = span;
    if (topView) {
      const yaw = azimuth - CONFIG.azimuth;
      camera.position.set(0, 0, cameraDistance);
      camera.up.set(-Math.sin(yaw), Math.cos(yaw), 0);
    } else {
      const radius = cameraDistance * Math.cos(elevation);
      camera.position.set(radius * Math.cos(azimuth), radius * Math.sin(azimuth), cameraDistance * Math.sin(elevation));
      camera.up.set(0, 0, 1);
    }
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
  }

  function updateView(now) {
    if (drag || now - lastInteraction < CONFIG.cameraIdleSeconds * 1000) return;
    if (azimuth === CONFIG.azimuth && elevation === CONFIG.elevation) return;
    if (!returnFrom) {
      const delta = Math.atan2(Math.sin(azimuth - CONFIG.azimuth), Math.cos(azimuth - CONFIG.azimuth));
      returnFrom = { azimuth: CONFIG.azimuth + delta, elevation, time: now };
    }
    const amount = ease((now - returnFrom.time) / (CONFIG.cameraReturnSeconds * 1000));
    azimuth = THREE.MathUtils.lerp(returnFrom.azimuth, CONFIG.azimuth, amount);
    elevation = THREE.MathUtils.lerp(returnFrom.elevation, CONFIG.elevation, amount);
    if (amount >= 1) returnFrom = null;
  }

  function updateIdleSpin(now) {
    if (idleSpinPeriodSeconds <= 0) return;
    const cameraAtDefault = azimuth === CONFIG.azimuth && elevation === CONFIG.elevation;
    if (drag || now - lastInteraction < CONFIG.cameraIdleSeconds * 1000 || !cameraAtDefault) {
      lastIdleSpinFrame = null;
      return;
    }
    if (lastIdleSpinFrame !== null) {
      const deltaSeconds = (now - lastIdleSpinFrame) / 1000;
      idleSpinAngle = (idleSpinAngle - 2 * Math.PI * deltaSeconds / idleSpinPeriodSeconds) % (2 * Math.PI);
    }
    lastIdleSpinFrame = now;
  }

  function render(now) {
    if (renderer.getContext().isContextLost()) {
      lastIdleSpinFrame = null;
      return;
    }
    updateView(now);
    updateIdleSpin(now);
    const size = canvas.getBoundingClientRect();
    if (size.width < 2 || size.height < 2) return;
    const outlinePixels = typeof outlineWidth === "function" ? outlineWidth(size) : outlineWidth;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = columns * Math.max(1, Math.round(size.width * pixelRatio / columns));
    const height = rows * Math.max(1, Math.round(size.height * pixelRatio / rows));
    const columnWidth = width / columns;
    const rowHeight = height / rows;
    const resized = renderer.domElement.width !== width || renderer.domElement.height !== height;
    if (resized) renderer.setSize(width, height, false);

    const viewChanged = resized || !previousView || previousView.azimuth !== azimuth || previousView.elevation !== elevation;
    const elapsed = (now - startTime) / 1000;
    const rotations = panels.map((panel) => {
      if (fixedAngle !== null) return { angle: fixedAngle, reference: 1, current: 1 };
      return animated ? rotationAt(elapsed, panel.fold) : { angle: idleSpinAngle, reference: 0, current: 0 };
    });
    if (!viewChanged && previousRotations && rotations.every((rotation, index) => {
      const previous = previousRotations[index];
      return previous.angle === rotation.angle && previous.reference === rotation.reference && previous.current === rotation.current;
    })) return;

    if (staggered) {
      renderer.setRenderTarget(null);
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, width, height);
      renderer.clear();
    }

    panels.forEach((panel, index) => {
      const rotation = rotations[index];
      const column = index % columns;
      const row = Math.floor(index / columns);
      const staggeredSize = Math.min(columnWidth, Math.round(height * 0.66));
      const panelWidth = staggered ? staggeredSize : columnWidth;
      const panelHeight = staggered ? staggeredSize : rowHeight;
      const x = column * columnWidth + (staggered && row === 1 ? columnWidth - panelWidth : 0);
      const y = staggered ? (row === 0 ? height - panelHeight : 0) : (rows - 1 - row) * panelHeight;
      if (!renderOutlines) {
        if (viewChanged) updateCamera(panel, panelWidth / panelHeight);
        panel.object.rotation.z = rotation.angle;
        renderer.setRenderTarget(null);
        renderer.setViewport(x, y, panelWidth, panelHeight);
        renderer.setScissor(x, y, panelWidth, panelHeight);
        renderer.setScissorTest(true);
        renderer.render(panel.scene, panel.camera);
        return;
      }
      if (viewChanged) {
        updateCamera(panel, panelWidth / panelHeight);
        if (panel.beauty.width !== panelWidth || panel.beauty.height !== panelHeight) {
          for (const target of [panel.beauty, panel.original, panel.rotated]) target.setSize(panelWidth, panelHeight);
        }
        if (rotation.reference > 0) {
          renderer.setScissorTest(false);
          panel.mask.rotation.z = 0;
          renderer.setRenderTarget(panel.original);
          renderer.render(panel.maskScene, panel.camera);
        }
      }

      const previous = previousRotations?.[index];
      const poseChanged = viewChanged || !previous || previous.angle !== rotation.angle;
      renderer.setScissorTest(false);
      if (poseChanged) {
        panel.object.rotation.z = rotation.angle;
        renderer.setRenderTarget(panel.beauty);
        renderer.render(panel.scene, panel.camera);
      }
      if (rotation.current > 0 && (poseChanged || !previous || previous.current === 0)) {
        panel.mask.rotation.z = rotation.angle;
        renderer.setRenderTarget(panel.rotated);
        renderer.render(panel.maskScene, panel.camera);
      }

      const { uniforms } = compositor;
      uniforms.beauty.value = panel.beauty.texture;
      uniforms.original.value = panel.original.texture;
      uniforms.rotated.value = panel.rotated.texture;
      uniforms.pixel.value.set(outlinePixels * pixelRatio / panelWidth, outlinePixels * pixelRatio / panelHeight);
      uniforms.referenceOpacity.value = rotation.reference;
      uniforms.currentOpacity.value = rotation.current;

      renderer.setRenderTarget(null);
      renderer.setViewport(x, y, panelWidth, panelHeight);
      renderer.setScissor(x, y, panelWidth, panelHeight);
      renderer.setScissorTest(true);
      if (staggered) renderer.autoClear = false;
      renderer.render(compositor.scene, compositor.camera);
      if (staggered) renderer.autoClear = true;
    });
    renderer.setScissorTest(false);
    previousView = { azimuth, elevation };
    previousRotations = rotations;
  }

  function tick(now) {
    render(now);
    frame = requestAnimationFrame(tick);
  }

  function scheduleStaticRender() {
    if (!staticRunning || frame !== null) return;
    frame = requestAnimationFrame((now) => {
      frame = null;
      render(now);
    });
  }

  // 海报只在首次显示或尺寸改变时绘制，不注册鼠标拖动事件。
  if (fixedAngle !== null) {
    const observer = new ResizeObserver(scheduleStaticRender);
    return {
      start() {
        if (staticRunning) return;
        staticRunning = true;
        observer.observe(canvas);
        scheduleStaticRender();
      },
      stop() {
        staticRunning = false;
        observer.disconnect();
        if (frame !== null) cancelAnimationFrame(frame);
        frame = null;
      },
    };
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const bounds = canvas.getBoundingClientRect();
    const column = clamp(Math.floor((event.clientX - bounds.left) / bounds.width * columns), 0, columns - 1);
    const row = clamp(Math.floor((event.clientY - bounds.top) / bounds.height * rows), 0, rows - 1);
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY, top: panels[row * columns + column].topView };
    lastInteraction = performance.now();
    returnFrom = null;
    lastIdleSpinFrame = null;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    const height = canvas.clientHeight || 1;
    const sensitivity = 2 * Math.PI / height;
    azimuth -= (event.clientX - drag.x) * sensitivity;
    if (!drag.top) elevation = clamp(elevation + (event.clientY - drag.y) * sensitivity, -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05);
    drag.x = event.clientX;
    drag.y = event.clientY;
    lastInteraction = performance.now();
  });
  function endDrag(event) {
    if (!drag || event.pointerId !== drag.id) return;
    drag = null;
    lastInteraction = performance.now();
  }
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("lostpointercapture", endDrag);

  return {
    start() {
      if (frame !== null) return;
      azimuth = CONFIG.azimuth;
      elevation = CONFIG.elevation;
      returnFrom = null;
      drag = null;
      lastInteraction = -Infinity;
      startTime = performance.now();
      idleSpinAngle = 0;
      lastIdleSpinFrame = null;
      previousView = null;
      frame = requestAnimationFrame(tick);
    },
    stop() {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      if (drag && canvas.hasPointerCapture(drag.id)) canvas.releasePointerCapture(drag.id);
      drag = null;
    },
  };
}
