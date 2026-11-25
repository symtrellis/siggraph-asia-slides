import * as THREE from "three";
import { GLTFLoader } from "../../vendor/three/examples/jsm/loaders/GLTFLoader.js";
import { createRotatingComparison } from "./spinning-top.js";

export async function createSectorGearScene(canvas) {
  const source = new URL("../../assets/models/01_gear_sector_replication.glb", import.meta.url);
  const gltf = await new GLTFLoader().loadAsync(source.href);
  const meshes = [];
  gltf.scene.traverse((object) => {
    if (object.isMesh) meshes.push(object);
  });
  if (meshes.length !== 1) throw new Error("齿轮模型的网格数量错误");
  const { geometry, material } = meshes[0];
  if (!material.isMeshStandardMaterial || material.map || material.metalnessMap || material.roughnessMap) {
    throw new Error("齿轮没有使用无贴图金属材质");
  }

  // 来自 Rebuttal 的 bad_00_direct_symmetry_postprocess_metadata.json。
  // 已确认原 OBJ 与本次使用的 GLB 具有完全相同的顶点坐标。
  const sourceAxis = new THREE.Vector3(-0.0401429757475853, 0.9991559386253357, 0.008723113685846329).normalize();
  const sourceCenter = new THREE.Vector3(0.015002076514065266, 0.0005054501234553754, 0.011143339797854424);
  const zAxis = new THREE.Vector3(0, 0, 1);
  const alignment = new THREE.Quaternion().setFromUnitVectors(sourceAxis, zAxis);
  geometry.translate(-sourceCenter.x, -sourceCenter.y, -sourceCenter.z);
  geometry.applyQuaternion(alignment);
  // 观察中心在真实主轴上，沿 Z 轴移动到齿轮高度的中间。
  geometry.computeBoundingBox();
  geometry.translate(0, 0, -(geometry.boundingBox.min.z + geometry.boundingBox.max.z) / 2);
  geometry.computeBoundingSphere();
  const fittingRadius = geometry.boundingSphere.radius + geometry.boundingSphere.center.length();

  return createRotatingComparison(canvas, {
    columns: 1,
    rows: 1,
    outlinePixels: 0,
    animated: false,
    idleSpinPeriodSeconds: 24,
    items: [{
      geometry, material, topView: false, fittingRadius,
      perspective: true,
      cameraDistance: 5.5 * fittingRadius,
    }],
  });
}
