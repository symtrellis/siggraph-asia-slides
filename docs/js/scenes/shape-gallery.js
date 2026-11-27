import * as THREE from "three";
import { GLTFLoader } from "../../vendor/three/examples/jsm/loaders/GLTFLoader.js";
import { createRotatingComparison } from "./spinning-top.js";

const assets = new URL("../../assets/models/gallery/", import.meta.url);
const examples = ["00_turboblade", "01_fountain", "02_icosahedral", "06_gear"];

async function loadModel(loader, entry) {
  const url = new URL(entry.file, assets);
  if (entry.output_sha256) url.searchParams.set("v", entry.output_sha256);
  const gltf = await loader.loadAsync(url.href);
  const meshes = [];
  gltf.scene.traverse((object) => {
    if (object.isMesh) meshes.push(object);
  });
  if (meshes.length !== 1) throw new Error(`模型网格数量错误：${entry.file}`);

  const { geometry, material } = meshes[0];
  if (!material.isMeshStandardMaterial) {
    throw new Error(`模型缺少 PBR 材质：${entry.file}`);
  }
  if (entry.textured && (!material.map || !material.metalnessMap || !material.roughnessMap)) {
    throw new Error(`模型缺少原始颜色、金属度或粗糙度贴图：${entry.file}`);
  }
  if (geometry.index.count !== entry.triangle_count * 3) {
    throw new Error(`网格三角面数量错误：${entry.file}`);
  }

  // 从 GLB 的 Y-up 坐标转换到用于对称轴检测的 Z-up 坐标。
  geometry.rotateX(Math.PI / 2);
  const center = new THREE.Vector3(...entry.axis_point_z_up);
  const axis = new THREE.Vector3(...entry.axis_z_up).normalize();
  const alignment = new THREE.Quaternion().setFromUnitVectors(axis, new THREE.Vector3(0, 0, 1));
  geometry.translate(-center.x, -center.y, -center.z);
  geometry.applyQuaternion(alignment);
  geometry.computeBoundingSphere();

  return { geometry, material, fold: entry.fold, topView: false };
}

export async function createShapeGallery(canvas, { outlinePixels = 0.82 } = {}) {
  const response = await fetch(new URL("manifest.json?v=compressed-pbr-final", assets));
  if (!response.ok) throw new Error(`对比模型参数读取失败：${response.status}`);
  const { models } = await response.json();
  const loader = new GLTFLoader();
  const upper = [];
  const lower = [];

  // 每次加载同一组的两份完整模型，减少同时加载网格的内存开销。
  for (const example of examples) {
    const vanilla = models.find((model) => model.example === example && model.variant === "vanilla");
    const symmetric = models.find((model) => model.example === example && model.variant === "symtrellis");
    if (!vanilla || !symmetric) throw new Error(`缺少对比模型：${example}`);
    const pair = await Promise.all([loadModel(loader, vanilla), loadModel(loader, symmetric)]);

    // 同一列使用共同的取景范围，整个旋转过程中保持等比例和固定旋转中心。
    const radius = Math.max(...pair.map(({ geometry }) =>
      geometry.boundingSphere.center.length() + geometry.boundingSphere.radius
    ));
    pair.forEach((model) => { model.fittingRadius = radius; });
    upper.push(pair[0]);
    lower.push(pair[1]);
  }

  return createRotatingComparison(canvas, {
    columns: examples.length,
    rows: 2,
    outlinePixels,
    items: [...upper, ...lower],
  });
}
