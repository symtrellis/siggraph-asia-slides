import * as THREE from "three";
import { GLTFLoader } from "../../vendor/three/examples/jsm/loaders/GLTFLoader.js";
import { EXRLoader } from "../../vendor/three/examples/jsm/loaders/EXRLoader.js";
import { MeshoptDecoder } from "../../vendor/three/examples/jsm/libs/meshopt_decoder.module.js";
import { rotationAt } from "./spinning-top.js";

const assetFolder = new URL("../../assets/models/fold-manipulation/", import.meta.url);
const environmentImage = new URL("../../assets/images/citrus_orchard_road_puresky_1k.exr", import.meta.url);
const zAxis = new THREE.Vector3(0, 0, 1);

// 00_gear.blend 的相机，减去原图中所有齿轮共同的 Z 方向 0.5 位移。
const cameraPosition = new THREE.Vector3(1.8394891024, 0.4567800164, 1.6624519825);
const cameraDirection = new THREE.Vector3(0.7652642727, 0.187966466, 0.6156616211).normalize();
const cameraDistance = cameraPosition.dot(cameraDirection);
const cameraTarget = cameraPosition.clone().addScaledVector(cameraDirection, -cameraDistance);
const cameraFov = THREE.MathUtils.radToDeg(2 * Math.atan(36 / (2 * 78)));
const initialAzimuth = Math.atan2(cameraDirection.y, cameraDirection.x);
const initialElevation = Math.asin(cameraDirection.z);
const ease = (value) => {
  const t = THREE.MathUtils.clamp(value, 0, 1);
  return t * t * t * (10 + t * (-15 + 6 * t));
};

async function loadFold(loader, entry, slot, environment, staticComparison) {
  const url = new URL(entry.file, assetFolder);
  url.searchParams.set("v", entry.output_sha256);
  const gltf = await loader.loadAsync(url.href);
  const meshes = [];
  gltf.scene.traverse((object) => { if (object.isMesh) meshes.push(object); });
  if (meshes.length !== 1) throw new Error(`齿轮网格数量错误：${entry.file}`);
  const mesh = meshes[0];
  if (mesh.geometry.index.count !== entry.triangles * 3 ||
      !mesh.material.isMeshStandardMaterial || !mesh.material.map ||
      !mesh.material.metalnessMap || !mesh.material.roughnessMap) {
    throw new Error(`齿轮几何或贴图不完整：${entry.file}`);
  }
  if (staticComparison) {
    // 纸质海报继续使用 00_gear.blend 原来的 Warm steel gray 材质。
    mesh.material = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setRGB(0.57758057, 0.59720218, 0.63075739),
      metalness: 1,
      roughness: 0.3,
      side: mesh.material.side,
    });
  }

  // 保留量化 GLB 的节点变换；不把浮点旋转写回整数顶点。
  const zUp = new THREE.Group();
  zUp.rotation.x = Math.PI / 2;
  zUp.add(gltf.scene);
  const axis = new THREE.Vector3(...entry.axis_z_up).normalize();
  const center = new THREE.Vector3(...entry.axis_point_z_up);
  const aligned = new THREE.Group();
  aligned.quaternion.setFromUnitVectors(axis, zAxis);
  aligned.position.copy(center).negate().applyQuaternion(aligned.quaternion);
  aligned.add(zUp);
  const rotor = new THREE.Group();
  rotor.add(aligned);
  const scene = new THREE.Scene();
  scene.add(rotor);
  scene.environment = environment.texture;
  scene.environmentRotation.x = Math.PI / 2;
  scene.environmentIntensity = 1;
  return { slot, scene, rotor, fold: entry.fold };
}

// 同一个画布按实际图片位置分别渲染，幻灯片和纸质海报均保留原来的标签布局。
export async function createFoldGallery(canvas, { staticComparison = false } = {}) {
  const slots = Array.from(canvas.parentElement.querySelectorAll("[data-fold-model]"));
  const response = await fetch(new URL("manifest.json", assetFolder));
  if (!response.ok) throw new Error(`齿轮模型清单读取失败：${response.status}`);
  const { models } = await response.json();
  const entries = new Map(models.map((entry) => [entry.fold, entry]));
  const renderer = new THREE.WebGLRenderer({
    canvas, alpha: true, antialias: true, powerPreference: "high-performance",
    preserveDrawingBuffer: staticComparison,
  });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.toneMappingExposure = 1;
  renderer.autoClear = false;
  const texture = await new EXRLoader().setDataType(THREE.HalfFloatType).loadAsync(environmentImage.href);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  function makeEnvironment() {
    const generator = new THREE.PMREMGenerator(renderer);
    const environment = generator.fromEquirectangular(texture);
    generator.dispose();
    return environment;
  }
  let environment = makeEnvironment();
  const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
  const panels = [];
  for (let i = 0; i < slots.length; i += 2) {
    const next = slots.slice(i, i + 2);
    panels.push(...await Promise.all(next.map((slot) => {
      const entry = entries.get(Number(slot.dataset.foldModel));
      if (!entry) throw new Error(`齿轮模型不存在：${slot.dataset.foldModel}`);
      return loadFold(loader, entry, slot, environment, staticComparison);
    })));
  }

  const camera = new THREE.PerspectiveCamera(cameraFov, 1, 0.1, 20);
  let active = false;
  let frame = null;
  let startTime = 0;
  let azimuth = initialAzimuth;
  let elevation = initialElevation;
  let drag = null;
  let lastInteraction = -Infinity;
  let returnFrom = null;
  let previous = "";

  function updateCamera(now) {
    if (!staticComparison && !drag && now - lastInteraction >= 5000 &&
        (azimuth !== initialAzimuth || elevation !== initialElevation)) {
      if (!returnFrom) {
        const delta = Math.atan2(Math.sin(azimuth - initialAzimuth), Math.cos(azimuth - initialAzimuth));
        returnFrom = { azimuth: initialAzimuth + delta, elevation, time: now };
      }
      const t = ease((now - returnFrom.time) / 900);
      azimuth = THREE.MathUtils.lerp(returnFrom.azimuth, initialAzimuth, t);
      elevation = THREE.MathUtils.lerp(returnFrom.elevation, initialElevation, t);
      if (t === 1) returnFrom = null;
    }
    camera.position.set(
      cameraDistance * Math.cos(elevation) * Math.cos(azimuth),
      cameraDistance * Math.cos(elevation) * Math.sin(azimuth),
      cameraDistance * Math.sin(elevation),
    ).add(cameraTarget);
    camera.up.copy(zAxis);
    camera.lookAt(cameraTarget);
    camera.updateMatrixWorld();
  }

  function draw(now) {
    if (!active || renderer.getContext().isContextLost()) return;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width < 2 || bounds.height < 2) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.max(1, Math.round(bounds.width * pixelRatio));
    const height = Math.max(1, Math.round(bounds.height * pixelRatio));
    const resized = canvas.width !== width || canvas.height !== height;
    if (resized) renderer.setSize(width, height, false);
    updateCamera(now);
    const angleValues = panels.map(({ fold }) =>
      staticComparison ? 0 : rotationAt((now - startTime) / 1000, fold).angle);
    const signature = [width, height, azimuth, elevation, ...angleValues].join(",");
    if (!resized && signature === previous) return;
    previous = signature;
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, width, height);
    renderer.clear(true, true, true);
    renderer.setScissorTest(true);
    panels.forEach(({ slot, scene, rotor }, index) => {
      const rect = slot.getBoundingClientRect();
      const square = Math.min(rect.width, rect.height);
      if (square < 2) return;
      const side = Math.max(1, Math.round(square * pixelRatio));
      const x = Math.round((rect.left - bounds.left + (rect.width - square) / 2) * pixelRatio);
      const y = Math.round((bounds.bottom - rect.bottom + (rect.height - square) / 2) * pixelRatio);
      rotor.rotation.z = angleValues[index];
      renderer.setViewport(x, y, side, side);
      renderer.setScissor(x, y, side, side);
      renderer.clear(true, true, true);
      renderer.render(scene, camera);
    });
    renderer.setScissorTest(false);
  }

  function tick(now) {
    frame = null;
    draw(now);
    if (active && !staticComparison) frame = requestAnimationFrame(tick);
  }

  function invalidate() {
    previous = "";
    if (active && frame === null) frame = requestAnimationFrame(tick);
  }

  const observer = new ResizeObserver(invalidate);
  canvas.addEventListener("webglcontextrestored", () => {
    environment.dispose();
    environment = makeEnvironment();
    panels.forEach(({ scene }) => { scene.environment = environment.texture; });
    invalidate();
  });

  if (!staticComparison) {
    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
      returnFrom = null;
      lastInteraction = performance.now();
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!drag || event.pointerId !== drag.id) return;
      const sensitivity = 2 * Math.PI / Math.max(1, canvas.clientHeight);
      azimuth -= (event.clientX - drag.x) * sensitivity;
      elevation = THREE.MathUtils.clamp(elevation + (event.clientY - drag.y) * sensitivity, 0.05, Math.PI / 2 - 0.05);
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
  }

  return {
    start() {
      if (active) return;
      active = true;
      startTime = performance.now();
      azimuth = initialAzimuth;
      elevation = initialElevation;
      lastInteraction = -Infinity;
      drag = null;
      returnFrom = null;
      observer.observe(canvas);
      invalidate();
    },
    stop() {
      active = false;
      observer.disconnect();
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      if (drag && canvas.hasPointerCapture(drag.id)) canvas.releasePointerCapture(drag.id);
      drag = null;
    },
  };
}
