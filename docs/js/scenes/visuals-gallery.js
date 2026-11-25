import * as THREE from "three";
import { GLTFLoader } from "../../vendor/three/examples/jsm/loaders/GLTFLoader.js";
import { createRotatingComparison } from "./spinning-top.js?v=visuals-no-outlines";

const assets = new URL("../../assets/models/visuals/", import.meta.url);

function symmetryFold(label) {
  // 采用项目生成时的主轴；二十面体沿五重轴旋转。
  if (label === "I" || label === "Ih") return 5;
  const match = /^[CD](\d+)/.exec(label);
  if (!match) throw new Error(`无法确定对称阶：${label}`);
  return Number(match[1]);
}

async function loadModel(loader, entry) {
  const url = new URL(entry.file, assets);
  url.searchParams.set("v", entry.output_sha256);
  const gltf = await loader.loadAsync(url.href);
  const meshes = [];
  gltf.scene.traverse((object) => {
    if (object.isMesh) meshes.push(object);
  });
  if (meshes.length !== 1) throw new Error(`模型网格数量错误：${entry.file}`);
  const { geometry, material } = meshes[0];
  if (geometry.index.count !== entry.triangle_count * 3 ||
      !material.isMeshStandardMaterial || !material.map ||
      !material.metalnessMap || !material.roughnessMap) {
    throw new Error(`模型几何或贴图错误：${entry.file}`);
  }
  // GLB 为 Y-up，生成设置中的主轴和对称中心为 Z-up。
  geometry.rotateX(Math.PI / 2);
  const center = new THREE.Vector3(...entry.symmetry.center);
  const axis = new THREE.Vector3(...entry.symmetry.majorAxis).normalize();
  geometry.translate(-center.x, -center.y, -center.z);
  geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(axis, new THREE.Vector3(0, 0, 1)));
  geometry.computeBoundingSphere();
  const fittingRadius = geometry.boundingSphere.center.length() + geometry.boundingSphere.radius;
  return { geometry, material, fittingRadius, fold: symmetryFold(entry.symmetry.label), topView: false };
}

export async function createVisualsGallery(canvas) {
  const response = await fetch(new URL("manifest.json", assets));
  if (!response.ok) throw new Error(`Visuals 模型清单读取失败：${response.status}`);
  const { models } = await response.json();
  if (models.length !== 8) throw new Error("Visuals 需要八个模型");
  const items = [];
  const loader = new GLTFLoader();
  // 保持清单中的模型顺序，每次只并行加载两份模型。
  for (let i = 0; i < models.length; i += 2) {
    items.push(...await Promise.all(models.slice(i, i + 2).map(entry => loadModel(loader, entry))));
  }
  return createRotatingComparison(canvas, {
    columns: 4,
    rows: 2,
    renderOutlines: false,
    items,
  });
}
