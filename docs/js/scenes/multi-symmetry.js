import * as THREE from "three";
import { GLTFLoader } from "../../vendor/three/examples/jsm/loaders/GLTFLoader.js";
import { EXRLoader } from "../../vendor/three/examples/jsm/loaders/EXRLoader.js";

const assets = new URL("../../assets/models/multi-symmetry/", import.meta.url);
const hdri = new URL("../../assets/images/symmetrical_garden_02_1k.exr", import.meta.url);
const idleSeconds = 5;
const returnSeconds = 0.9;
const defaultZoom = 1.15;

export async function createMultiSymmetryScene(canvas) {
  const loader = new GLTFLoader();
  const [manifest, segmentation, chair, texture] = await Promise.all([
    fetch(new URL("manifest.json", assets)).then(response => {
      if (!response.ok) throw new Error("Multi-Symmetry 模型清单读取失败");
      return response.json();
    }),
    loader.loadAsync(new URL("space-segmentation.glb", assets).href),
    loader.loadAsync(new URL("symtrellis-chair.glb", assets).href),
    new EXRLoader().setDataType(THREE.HalfFloatType).loadAsync(hdri.href),
  ]);
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.toneMappingExposure = 1;
  texture.mapping = THREE.EquirectangularReflectionMapping;
  function makeEnvironment() {
    const generator = new THREE.PMREMGenerator(renderer);
    const result = generator.fromEquirectangular(texture);
    generator.dispose();
    return result;
  }
  let environment = makeEnvironment();

  let voxelMeshes = 0;
  let regionMeshes = 0;
  segmentation.scene.traverse(object => {
    if (!object.isMesh) return;
    if (object.geometry.index.count / 3 === 83748) {
      voxelMeshes++;
      // Blender 中的体素椅子没有材质槽，采用默认漫反射白模。
      object.material = new THREE.MeshStandardMaterial({
        color: new THREE.Color().setRGB(0.8, 0.8, 0.8),
        roughness: 0.5, metalness: 0, flatShading: true, side: THREE.DoubleSide,
      });
    } else {
      regionMeshes++;
      // GLB 包含 Blender 的真实尺寸及蓝/绿色、0.25 透明度。
      object.material.transparent = true;
      object.material.depthWrite = false;
      object.material.side = THREE.DoubleSide;
    }
  });
  let outputMeshes = 0;
  chair.scene.traverse(object => {
    if (!object.isMesh) return;
    if (object.geometry.index.count / 3 !== 470056) throw new Error("SymTRELLIS 椅子网格不正确");
    outputMeshes++;
  });
  if (voxelMeshes !== 1 || regionMeshes !== 2 || outputMeshes !== 1) {
    throw new Error("Multi-Symmetry 场景的对象数量不正确");
  }

  const scenes = [segmentation.scene, chair.scene].map(model => {
    const scene = new THREE.Scene();
    // Blender 原始坐标系 Z-up；GLB 导出后采用 Y-up。
    model.rotation.x = Math.PI / 2;
    scene.add(model);
    scene.environment = environment.texture;
    scene.environmentRotation.x = Math.PI / 2;
    scene.environmentIntensity = 1;
    return scene;
  });
  const position = new THREE.Vector3(...manifest.camera.position);
  const distance = position.length();
  const defaultAzimuth = Math.atan2(position.y, position.x);
  const defaultElevation = Math.asin(position.z / distance);
  const camera = new THREE.PerspectiveCamera(manifest.camera.fov_degrees, 1, manifest.camera.near, manifest.camera.far);
  let azimuth = defaultAzimuth;
  let elevation = defaultElevation;
  let zoom = 1;
  let targetZoom = 1;
  let frame = null;
  let lastFrame = null;
  let lastInteraction = -Infinity;
  let drag = null;
  let returning = null;
  let previous = null;

  function cancelReturn() {
    if (!returning) return;
    targetZoom = zoom;
    returning = null;
  }

  function updateView(now) {
    const elapsed = lastFrame === null ? 0 : Math.min((now - lastFrame) / 1000, 0.1);
    lastFrame = now;
    if (!returning) {
      zoom += (targetZoom - zoom) * (1 - Math.exp(-elapsed / 0.075));
      if (Math.abs(zoom - targetZoom) < 0.0001) zoom = targetZoom;
    }
    if (drag || now - lastInteraction < idleSeconds * 1000) return;
    if (!returning && (azimuth !== defaultAzimuth || elevation !== defaultElevation || zoom !== 1)) {
      const difference = Math.atan2(Math.sin(azimuth - defaultAzimuth), Math.cos(azimuth - defaultAzimuth));
      returning = { azimuth: defaultAzimuth + difference, elevation, zoom, time: now };
      targetZoom = 1;
    }
    if (!returning) return;
    const t = THREE.MathUtils.clamp((now - returning.time) / (returnSeconds * 1000), 0, 1);
    const ease = t * t * t * (10 + t * (-15 + 6 * t));
    azimuth = THREE.MathUtils.lerp(returning.azimuth, defaultAzimuth, ease);
    elevation = THREE.MathUtils.lerp(returning.elevation, defaultElevation, ease);
    zoom = THREE.MathUtils.lerp(returning.zoom, 1, ease);
    if (t >= 1) {
      azimuth = defaultAzimuth;
      elevation = defaultElevation;
      zoom = 1;
      returning = null;
    }
  }

  function render(now) {
    if (renderer.getContext().isContextLost()) return;
    updateView(now);
    const size = canvas.getBoundingClientRect();
    if (size.width < 2 || size.height < 2) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    const panelWidth = Math.max(1, Math.round(size.width * pixelRatio / 2));
    const height = Math.max(1, Math.round(size.height * pixelRatio));
    const width = panelWidth * 2;
    const resized = canvas.width !== width || canvas.height !== height;
    if (resized) renderer.setSize(width, height, false);
    if (!resized && previous?.azimuth === azimuth && previous?.elevation === elevation &&
        previous?.zoom === zoom) return;

    const horizontal = distance * Math.cos(elevation);
    camera.position.set(horizontal * Math.cos(azimuth), horizontal * Math.sin(azimuth), distance * Math.sin(elevation));
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, 0);
    camera.aspect = panelWidth / height;
    camera.zoom = defaultZoom * zoom;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    renderer.setScissorTest(true);
    scenes.forEach((scene, index) => {
      const x = index * panelWidth;
      renderer.setViewport(x, 0, panelWidth, height);
      renderer.setScissor(x, 0, panelWidth, height);
      renderer.render(scene, camera);
    });
    renderer.setScissorTest(false);
    previous = { azimuth, elevation, zoom };
  }

  function tick(now) {
    render(now);
    frame = requestAnimationFrame(tick);
  }

  canvas.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    cancelReturn();
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
    lastInteraction = performance.now();
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", event => {
    if (!drag || event.pointerId !== drag.id) return;
    const sensitivity = 2 * Math.PI / Math.max(canvas.clientHeight, 1);
    azimuth -= (event.clientX - drag.x) * sensitivity;
    elevation = THREE.MathUtils.clamp(elevation + (event.clientY - drag.y) * sensitivity, -Math.PI / 2 + 0.05, Math.PI / 2 - 0.05);
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
  canvas.addEventListener("wheel", event => {
    if (canvas.closest(".slide").hidden) return;
    event.preventDefault();
    cancelReturn();
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1);
    targetZoom = THREE.MathUtils.clamp(targetZoom * Math.exp(-delta * 0.003), 0.65, 3);
    lastInteraction = performance.now();
  }, { passive: false });
  canvas.addEventListener("webglcontextrestored", () => {
    environment.dispose();
    environment = makeEnvironment();
    scenes.forEach(scene => { scene.environment = environment.texture; });
    previous = null;
  });

  return {
    start() {
      if (frame !== null) return;
      azimuth = defaultAzimuth;
      elevation = defaultElevation;
      zoom = targetZoom = 1;
      drag = returning = previous = lastFrame = null;
      lastInteraction = -Infinity;
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
