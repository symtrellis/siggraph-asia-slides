import * as THREE from "three";
import { GLTFLoader } from "../../vendor/three/examples/jsm/loaders/GLTFLoader.js";
import { EXRLoader } from "../../vendor/three/examples/jsm/loaders/EXRLoader.js";

const modelFolder = new URL("../../assets/models/dragon-comparison/", import.meta.url);
const hdriSource = new URL("../../assets/images/ludwikowice_farmland_1k.exr", import.meta.url);
const poseComparisonFiles = [
  "active-original.glb", "active-rotated.glb", "active-nearest.glb",
  "shape-original.glb", "shape-rotated.glb", "shape-nearest.glb",
];
const mapperComparisonFiles = [
  "active-nearest.glb", "active-mapped.glb",
  "shape-nearest.glb", "shape-mapped.glb",
];

// 03_mapper_results.blend 中实际使用的相机矩阵、镜头和传感器宽度。
const cameraMatrix = new THREE.Matrix4().set(
  0.8004128933, 0.5904060602, -0.1037297845, -0.1798264086,
  -0.5714879036, 0.6993334889, -0.4293416440, -0.5478200316,
  -0.1809442192, 0.4029309154, 0.8971654773, 1.1837357283,
  0, 0, 0, 1,
);
// Blender 原来的方形画面取景较紧。完整环绕时纵向最大投影是原画面的 1.285 倍；
// 六个模型统一放宽取景以保留边缘余量，相机的位置和朝向仍完全来自 Blender。
const orbitFramingScale = 1.37;
const cameraFov = THREE.MathUtils.radToDeg(2 * Math.atan(orbitFramingScale * 36 / (2 * 51)));
const basePosition = new THREE.Vector3().setFromMatrixPosition(cameraMatrix);
const baseOrientation = new THREE.Quaternion().setFromRotationMatrix(cameraMatrix).normalize();
const baseBack = new THREE.Vector3(0, 0, 1).applyQuaternion(baseOrientation);
const baseRight = new THREE.Vector3(1, 0, 0).applyQuaternion(baseOrientation);
const pivot = basePosition.clone().addScaledVector(baseBack, -basePosition.dot(baseBack));
const baseOffset = basePosition.clone().sub(pivot);
const zAxis = new THREE.Vector3(0, 0, 1);
const orbitPeriod = 24;
const idleSeconds = 5;
const returnSeconds = 0.9;

// 原 Blender 材质 SpatialVariantA：按固定世界坐标，对八个角的线性 RGB 插值。
// 不把每个模型分别重新归一化，避免改变不同姿态的颜色含义。
function makeSpatialMaterial(flatShading) {
  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff, metalness: 0, roughness: 0.4, flatShading, side: THREE.DoubleSide,
  });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 spatialPosition;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nspatialPosition = position;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>
        varying vec3 spatialPosition;
        vec3 spatialColor(vec3 position) {
          vec3 p = clamp((position - vec3(-0.46875, -0.3125, -0.21875)) /
            vec3(0.9375, 0.625, 0.4375), 0.0, 1.0);
          vec3 a = mix(vec3(0.20, 0.10, 0.42), vec3(0.92, 0.28, 0.55), p.x);
          vec3 b = mix(vec3(0.18, 0.32, 0.65), vec3(0.98, 0.58, 0.20), p.x);
          vec3 c = mix(vec3(0.15, 0.58, 0.68), vec3(0.98, 0.68, 0.55), p.x);
          vec3 d = mix(vec3(0.22, 0.62, 0.50), vec3(1.00, 0.92, 0.62), p.x);
          return mix(mix(a, b, p.y), mix(c, d, p.y), p.z);
        }
      `)
      .replace("#include <color_fragment>", "#include <color_fragment>\ndiffuseColor.rgb *= spatialColor(spatialPosition);");
  };
  return material;
}

async function loadDragon(loader, file, material) {
  const gltf = await loader.loadAsync(new URL(file, modelFolder).href);
  const meshes = [];
  gltf.scene.traverse((object) => { if (object.isMesh) meshes.push(object); });
  if (meshes.length !== 1) throw new Error(`龙模型网格数量错误：${file}`);
  const source = meshes[0];
  const geometry = source.geometry;
  if (geometry.attributes.color || source.material.map) throw new Error(`模型仍含有原始颜色：${file}`);
  // GLB 的 Y-up 对应 Blender 的 Z-up；保持原始位置和方向，不居中或自转网格。
  geometry.rotateX(Math.PI / 2);
  if (!material.flatShading && !geometry.attributes.normal) throw new Error(`缺少平滑法线：${file}`);
  geometry.computeBoundingSphere();
  return new THREE.Mesh(geometry, material);
}

export async function createDragonComparison(canvas, {
  staticComparison = false, combinedComparison = false, interactionContainer,
} = {}) {
  const isMapperComparison = canvas.hasAttribute("data-mapper-comparison");
  const files = combinedComparison
    ? [...poseComparisonFiles.slice(0, 3), mapperComparisonFiles[1],
       ...poseComparisonFiles.slice(3), mapperComparisonFiles[3]]
    : isMapperComparison ? mapperComparisonFiles : poseComparisonFiles;
  const columns = files.length / 2;
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: staticComparison });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.toneMappingExposure = staticComparison ? 0.82 : 1;

  const texture = await new EXRLoader().setDataType(THREE.HalfFloatType).loadAsync(hdriSource.href);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  function makeEnvironment() {
    const generator = new THREE.PMREMGenerator(renderer);
    const target = generator.fromEquirectangular(texture);
    generator.dispose();
    return target;
  }
  let environment = makeEnvironment();
  const loader = new GLTFLoader();
  const flat = makeSpatialMaterial(true);
  const smooth = makeSpatialMaterial(false);
  const objects = await Promise.all(files.map((file) => loadDragon(loader, file,
    file === "shape-nearest.glb" || file === "shape-mapped.glb" ? smooth : flat)));
  const scenes = objects.map((object) => {
    const scene = new THREE.Scene();
    scene.add(object);
    scene.environment = environment.texture;
    scene.environmentRotation.x = Math.PI / 2;
    scene.environmentIntensity = 1;
    return scene;
  });

  const camera = new THREE.PerspectiveCamera(cameraFov, 1, 0.1, 1000);
  const defaultZoom = isMapperComparison ? orbitFramingScale / 1.23 : 1;
  let zoomFactor = 1;
  let targetZoomFactor = 1;
  const yawRotation = new THREE.Quaternion();
  const pitchRotation = new THREE.Quaternion();
  const viewRotation = new THREE.Quaternion();
  let frame = null;
  let automaticYaw = 0;
  let yaw = 0;
  let pitch = 0;
  let drag = null;
  let lastInteraction = -Infinity;
  let returnFrom = null;
  let lastFrame = null;
  let previousView = null;

  function updateCamera() {
    yawRotation.setFromAxisAngle(zAxis, yaw);
    pitchRotation.setFromAxisAngle(baseRight, -pitch);
    viewRotation.copy(yawRotation).multiply(pitchRotation);
    camera.position.copy(baseOffset).applyQuaternion(viewRotation).add(pivot);
    camera.quaternion.copy(viewRotation).multiply(baseOrientation);
    camera.updateMatrixWorld();
  }

  function updateView(now) {
    const delta = lastFrame === null ? 0 : Math.min((now - lastFrame) / 1000, 0.1);
    lastFrame = now;
    if (!returnFrom) {
      zoomFactor += (targetZoomFactor - zoomFactor) * (1 - Math.exp(-delta / 0.075));
      if (Math.abs(targetZoomFactor - zoomFactor) < 0.0001) zoomFactor = targetZoomFactor;
    }
    if (drag || now - lastInteraction < idleSeconds * 1000) return;

    if (!returnFrom && (yaw !== automaticYaw || pitch !== 0 || zoomFactor !== 1)) {
      const difference = Math.atan2(Math.sin(yaw - automaticYaw), Math.cos(yaw - automaticYaw));
      returnFrom = { yaw: automaticYaw + difference, pitch, zoomFactor, time: now };
      targetZoomFactor = 1;
    }
    if (returnFrom) {
      const t = THREE.MathUtils.clamp((now - returnFrom.time) / (returnSeconds * 1000), 0, 1);
      const ease = t * t * t * (10 + t * (-15 + 6 * t));
      yaw = THREE.MathUtils.lerp(returnFrom.yaw, automaticYaw, ease);
      pitch = THREE.MathUtils.lerp(returnFrom.pitch, 0, ease);
      zoomFactor = THREE.MathUtils.lerp(returnFrom.zoomFactor, 1, ease);
      if (t >= 1) {
        yaw = automaticYaw;
        pitch = 0;
        zoomFactor = 1;
        returnFrom = null;
      }
      return;
    }
    automaticYaw = (automaticYaw + delta * 2 * Math.PI / orbitPeriod) % (2 * Math.PI);
    yaw = automaticYaw;
  }

  function draw(now) {
    if (renderer.getContext().isContextLost()) return;
    if (!staticComparison) updateView(now);
    const size = canvas.getBoundingClientRect();
    if (size.width < 2 || size.height < 2) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    const panelWidth = Math.max(1, Math.round(size.width * pixelRatio / columns));
    const panelHeight = Math.max(1, Math.round(size.height * pixelRatio / 2));
    const width = panelWidth * columns;
    const height = panelHeight * 2;
    const resized = canvas.width !== width || canvas.height !== height;
    if (resized) renderer.setSize(width, height, false);
    if (!resized && previousView?.yaw === yaw && previousView?.pitch === pitch && previousView?.zoomFactor === zoomFactor) return;

    // 所有窗口使用相同纵向视野和透视参数，较宽的窗口自然扩展横向视野。
    camera.aspect = panelWidth / panelHeight;
    const aspectFit = combinedComparison ? Math.min(1.12, camera.aspect * 1.08) : 1;
    camera.zoom = defaultZoom * zoomFactor * aspectFit;
    camera.updateProjectionMatrix();
    if (isMapperComparison) {
      // 四个模型统一使用更紧的取景，补偿完整环绕时纵向投影范围的不对称。
      camera.projectionMatrix.elements[9] -= 0.09;
      camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    }
    updateCamera();
    renderer.setScissorTest(true);
    scenes.forEach((scene, index) => {
      const x = index % columns * panelWidth;
      const y = (1 - Math.floor(index / columns)) * panelHeight;
      renderer.setViewport(x, y, panelWidth, panelHeight);
      renderer.setScissor(x, y, panelWidth, panelHeight);
      renderer.render(scene, camera);
    });
    renderer.setScissorTest(false);
    previousView = { yaw, pitch, zoomFactor };
  }

  function tick(now) {
    draw(now);
    frame = requestAnimationFrame(tick);
  }

  // 海报使用统一的较低斜视角；所有模型保持原始坐标和相对尺度。
  if (staticComparison) {
    let active = false;
    let retries = 0;
    const combinedBounds = new THREE.Box3();
    for (const object of objects) {
      object.geometry.computeBoundingBox();
      combinedBounds.union(object.geometry.boundingBox);
    }
    const center = combinedBounds.getCenter(new THREE.Vector3());
    const azimuth = THREE.MathUtils.degToRad(-75);
    const elevation = THREE.MathUtils.degToRad(27);
    const direction = new THREE.Vector3(Math.cos(azimuth) * Math.cos(elevation), Math.sin(azimuth) * Math.cos(elevation), Math.sin(elevation));
    const posterCamera = new THREE.PerspectiveCamera(22, 1, 0.02, 40);
    const posterRight = new THREE.Vector3().crossVectors(zAxis, direction).normalize();
    const posterUp = new THREE.Vector3().crossVectors(direction, posterRight).normalize();
    // 真实顶点的投影边界比包围盒更紧，仍然只计算一个供所有模型共享的相机距离。
    const projectedPositions = objects.map((object) => {
      const vertices = object.geometry.attributes.position;
      const result = new Float32Array(vertices.count * 3);
      const delta = new THREE.Vector3();
      for (let index = 0; index < vertices.count; index++) {
        delta.fromBufferAttribute(vertices, index).sub(center);
        result[index * 3] = Math.abs(delta.dot(posterRight));
        result[index * 3 + 1] = Math.abs(delta.dot(posterUp));
        result[index * 3 + 2] = delta.dot(direction);
      }
      return result;
    });
    function fitPosterCamera(aspect) {
      posterCamera.aspect = aspect;
      posterCamera.updateProjectionMatrix();
      posterCamera.up.copy(zAxis);
      const halfFov = Math.tan(THREE.MathUtils.degToRad(posterCamera.fov / 2)) * 0.94;
      let distance = 0.8;
      for (const positions of projectedPositions) {
        for (let index = 0; index < positions.length; index += 3) {
          distance = Math.max(distance, positions[index + 2] + Math.max(
            positions[index] / (halfFov * aspect), positions[index + 1] / halfFov,
          ));
        }
      }
      posterCamera.position.copy(center).addScaledVector(direction, distance);
      posterCamera.lookAt(center);
      posterCamera.updateMatrixWorld();
    }
    function drawStatic() {
      frame = null;
      if (!active || renderer.getContext().isContextLost()) return;
      const size = canvas.getBoundingClientRect();
      if (size.width < 2 || size.height < 2) return;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
      const panelWidth = Math.max(1, Math.round(size.width * pixelRatio / columns));
      const panelHeight = Math.max(1, Math.round(size.height * pixelRatio / 2));
      const width = columns * panelWidth;
      const height = 2 * panelHeight;
      if (canvas.width !== width || canvas.height !== height) renderer.setSize(width, height, false);
      fitPosterCamera(panelWidth / panelHeight);
      renderer.autoClear = false;
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, width, height);
      renderer.clear(true, true, true);
      renderer.setScissorTest(true);
      scenes.forEach((scene, index) => {
        const x = index % columns * panelWidth;
        const y = (1 - Math.floor(index / columns)) * panelHeight;
        renderer.setViewport(x, y, panelWidth, panelHeight);
        renderer.setScissor(x, y, panelWidth, panelHeight);
        renderer.render(scene, posterCamera);
      });
      renderer.setScissorTest(false);
      const gl = renderer.getContext();
      gl.finish();
      // 首帧着色器或网格尚未完成绘制时重试；正常情况下不持续渲染海报。
      const pixels = new Uint8Array(width * height * 4);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      const occupied = new Array(scenes.length).fill(false);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (pixels[(y * width + x) * 4 + 3] > 10) {
            const index = (y >= panelHeight ? 0 : columns) + Math.floor(x / panelWidth);
            occupied[index] = true;
          }
        }
      }
      if (!occupied.every(Boolean) && retries < 6) {
        retries++;
        frame = requestAnimationFrame(drawStatic);
      }
    }
    function invalidate() {
      if (!active || frame !== null) return;
      retries = 0;
      frame = requestAnimationFrame(drawStatic);
    }
    const observer = new ResizeObserver(invalidate);
    canvas.addEventListener("webglcontextrestored", () => {
      environment.dispose();
      environment = makeEnvironment();
      scenes.forEach((scene) => { scene.environment = environment.texture; });
      previousView = null;
      invalidate();
    });
    return {
      start() {
        if (active) return;
        active = true;
        previousView = null;
        observer.observe(canvas);
        invalidate();
      },
      stop() {
        active = false;
        observer.disconnect();
        if (frame !== null) cancelAnimationFrame(frame);
        frame = null;
      },
    };
  }

  function cancelReturn() {
    if (!returnFrom) return;
    targetZoomFactor = zoomFactor;
    returnFrom = null;
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
    lastInteraction = performance.now();
    cancelReturn();
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    const sensitivity = 2 * Math.PI / Math.max(canvas.clientHeight, 1);
    yaw -= (event.clientX - drag.x) * sensitivity;
    pitch = THREE.MathUtils.clamp(pitch + (event.clientY - drag.y) * sensitivity, -1.8, 0.35);
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
  // 两页的鼠标滚轮与 Chromium 触摸板捏合共用相机缩放。
  const slide = interactionContainer ?? canvas.closest(".slide");
  slide.addEventListener("wheel", (event) => {
    if (slide.hidden) return;
    event.preventDefault();
    cancelReturn();
    const deltaPixels = event.deltaY * (
      event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1
    );
    targetZoomFactor = THREE.MathUtils.clamp(
      targetZoomFactor * Math.exp(-deltaPixels * 0.003), 0.65, 3
    );
    lastInteraction = performance.now();
  }, { passive: false });
  canvas.addEventListener("webglcontextrestored", () => {
    environment.dispose();
    environment = makeEnvironment();
    scenes.forEach((scene) => { scene.environment = environment.texture; });
    previousView = null;
  });

  return {
    start() {
      if (frame !== null) return;
      automaticYaw = 0;
      yaw = 0;
      pitch = 0;
      zoomFactor = 1;
      targetZoomFactor = 1;
      drag = null;
      returnFrom = null;
      lastInteraction = -Infinity;
      lastFrame = null;
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
