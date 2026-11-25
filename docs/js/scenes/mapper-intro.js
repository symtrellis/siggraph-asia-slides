import * as THREE from "three";
import { STLLoader } from "../../vendor/three/examples/jsm/loaders/STLLoader.js";
import { EXRLoader } from "../../vendor/three/examples/jsm/loaders/EXRLoader.js";

const modelFolder = new URL("../../assets/models/", import.meta.url);
const studioImage = new URL("../../assets/images/neutral_studio.exr", import.meta.url);

// SymTRELLIS 陀螺的前侧旋臂，轴对齐后真实网格第 436258 个三角形的中心。
// 后续放大、体素化和邻居演示继续使用同一个物体坐标点。
export const mapperFocus = Object.freeze({
  position: Object.freeze([0.31488494, -0.64737074, 0.07417447]),
  innerRadius: 0.12,
  outerRadius: 0.27,
});

// 蓝色目标体素的网格索引在逆向回转之前固定，邻居在回转后的共同空间中检索。
export const mapperNeighborhood = Object.freeze({
  targetCell: Object.freeze([22, 25, 17]),
  radiusInCells: 2,
});

const darkMetal = new THREE.Color(0x454a50);
const brightMetal = new THREE.Color(0xc6ccd2);
const highlightMilliseconds = 350;
const splitMilliseconds = 700;
const rotationMilliseconds = 1000;
const voxelizeMilliseconds = 1100;
const gridAppearMilliseconds = 380;
const alignMilliseconds = 1000;
const localZoomMilliseconds = 1000;
const skeletonMilliseconds = 750;
const selectMilliseconds = 450;
const sharedViewMilliseconds = 120;
const mergeMilliseconds = 900;
const sphereMilliseconds = 500;
const splitEnd = highlightMilliseconds + splitMilliseconds;
const rotationEnd = splitEnd + rotationMilliseconds;
const voxelizeEnd = rotationEnd + voxelizeMilliseconds;
const alignEnd = voxelizeEnd + alignMilliseconds;
const localZoomEnd = alignEnd + localZoomMilliseconds;
const skeletonEnd = localZoomEnd + skeletonMilliseconds;
const selectEnd = skeletonEnd + selectMilliseconds;
const mergeStart = selectEnd + sharedViewMilliseconds;
const mergeEnd = mergeStart + mergeMilliseconds;
const sphereEnd = mergeEnd + sphereMilliseconds;
const stageTimes = [0, highlightMilliseconds, rotationEnd, voxelizeEnd, alignEnd, localZoomEnd, skeletonEnd, selectEnd, sphereEnd];
const sourceColor = new THREE.Color(0xf06762);
const rotatedColor = new THREE.Color(0x60aef2);
const contextGray = new THREE.Color(0x777777);

function ease(value) {
  const t = THREE.MathUtils.clamp(value, 0, 1);
  return t * t * t * (10 + t * (-15 + 6 * t));
}

async function loadSpinningTop() {
  const response = await fetch(new URL("rotation_axes.json", modelFolder));
  if (!response.ok) throw new Error("陀螺旋转轴读取失败");
  const { models } = await response.json();
  const model = models.symtrellis;
  const normalFile = model.source.replace(/\.stl$/i, ".normals.f32");
  const [geometry, normalResponse] = await Promise.all([
    new STLLoader().loadAsync(new URL(model.source, modelFolder).href),
    fetch(new URL(normalFile, modelFolder)),
  ]);
  if (!normalResponse.ok) throw new Error("陀螺平滑法线读取失败");
  const normals = new Float32Array(await normalResponse.arrayBuffer());
  if (normals.length !== geometry.attributes.position.array.length) throw new Error("陀螺顶点与法线数量不同");
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  // 与第四页保持相同的模型坐标和已经检测出的对称轴。
  const center = new THREE.Vector3(...model.axis_point_at_z0);
  const axis = new THREE.Vector3(...model.axis_direction).normalize();
  geometry.translate(-center.x, -center.y, -center.z);
  geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(axis, new THREE.Vector3(0, 0, 1)));
  geometry.normalizeNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

async function loadVoxelGrid() {
  const response = await fetch(new URL("mapper-voxels-32.json", modelFolder));
  if (!response.ok) throw new Error("体素占用缓存读取失败");
  const grid = await response.json();
  if (grid.resolution !== 32 || grid.rotation_degrees !== 120 || grid.source.length === 0 || grid.rotated.length === 0) {
    throw new Error("体素缓存不符合共同坐标系和旋转角度");
  }
  return grid;
}

function voxelWaveFunction(grid) {
  return `
    float voxelWave(float z) {
      float depth = clamp((${grid.lower[2] + grid.side} - z) / ${grid.side}, 0.0, 1.0);
      float start = 0.78 * depth;
      return smoothstep(start, start + 0.22, voxelizeAmount);
    }
  `;
}

function cellCenter(cell, grid) {
  return new THREE.Vector3(...cell.map((coord, axis) => grid.lower[axis] + (coord + 0.5) * grid.cell_size));
}

function selectMapperNeighborhood(grid) {
  const targetKey = mapperNeighborhood.targetCell.join(",");
  if (!grid.rotated.some((cell) => cell.join(",") === targetKey)) throw new Error("选定的蓝色体素不存在");
  const targetPosition = cellCenter(mapperNeighborhood.targetCell, grid)
    .applyAxisAngle(new THREE.Vector3(0, 0, 1), -2 * Math.PI / 3);
  const radius = mapperNeighborhood.radiusInCells * grid.cell_size;
  const sourceCells = grid.source.filter((cell) => cellCenter(cell, grid).distanceTo(targetPosition) <= radius + 1e-8);
  return {
    targetPosition, radius, sourceCells,
    sourceKeys: new Set(sourceCells.map((cell) => cell.join(","))),
    targetKeys: new Set([targetKey]),
  };
}

function createMetalMaterial(progress, voxelize, grid) {
  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff, metalness: 0.94, roughness: 0.26, side: THREE.DoubleSide, alphaHash: true,
  });
  material.onBeforeCompile = (shader) => {
    shader.uniforms.highlightAmount = progress;
    shader.uniforms.voxelizeAmount = voxelize;
    shader.uniforms.focusPosition = { value: new THREE.Vector3(...mapperFocus.position) };
    shader.uniforms.darkMetal = { value: darkMetal };
    shader.uniforms.brightMetal = { value: brightMetal };
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 mapperPosition;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nmapperPosition = position;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>
        varying vec3 mapperPosition;
        uniform float highlightAmount;
        uniform float voxelizeAmount;
        uniform vec3 focusPosition, darkMetal, brightMetal;
        ${voxelWaveFunction(grid)}
      `)
      .replace("#include <color_fragment>", `#include <color_fragment>
        float distanceToFocus = distance(mapperPosition, focusPosition);
        float neighborhood = 1.0 - smoothstep(${mapperFocus.innerRadius}, ${mapperFocus.outerRadius}, distanceToFocus);
        diffuseColor.rgb *= mix(darkMetal, brightMetal, neighborhood * highlightAmount);
        // 每一高度的原始表面随着对应体素长大而逐渐消失。
        diffuseColor.a *= 1.0 - smoothstep(0.35, 0.98, voxelWave(mapperPosition.z));
      `);
  };
  return material;
}

function createVoxelMaterial(voxelize, shrink, grid, tint, focus, dim, context) {
  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff, metalness: 0.88, roughness: 0.34, transparent: context,
  });
  material.onBeforeCompile = (shader) => {
    shader.uniforms.voxelizeAmount = voxelize;
    shader.uniforms.shrinkAmount = shrink;
    shader.uniforms.gridTint = { value: tint };
    shader.uniforms.gridFocus = { value: focus };
    shader.uniforms.dimAmount = dim;
    shader.uniforms.contextGray = { value: contextGray };
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>
        uniform float voxelizeAmount;
        uniform float shrinkAmount;
        uniform vec3 gridFocus;
        ${voxelWaveFunction(grid)}
      `)
      .replace("#include <begin_vertex>", `#include <begin_vertex>
        // 实例矩阵始终保存统一坐标系下的体素中心；GPU 按高度控制每个小立方体的大小。
        float localDistance = distance(instanceMatrix[3].xyz, gridFocus);
        float keepLocal = step(localDistance, ${mapperFocus.outerRadius + grid.cell_size / 2});
        transformed *= voxelWave(instanceMatrix[3].z) * mix(1.0, 0.17 * keepLocal, shrinkAmount);
      `);
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>
        uniform float shrinkAmount;
        uniform float dimAmount;
        uniform vec3 gridTint;
        uniform vec3 contextGray;
      `)
      .replace("#include <color_fragment>", `#include <color_fragment>
        diffuseColor.rgb = mix(mix(diffuseColor.rgb, gridTint, shrinkAmount), contextGray, dimAmount);
      `)
      .replace("#include <metalnessmap_fragment>", "#include <metalnessmap_fragment>\nmetalnessFactor = mix(metalnessFactor, 0.15, shrinkAmount);")
      .replace("#include <roughnessmap_fragment>", "#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, 0.5, shrinkAmount);");
  };
  return material;
}

function makeVoxelMesh(cells, focus, grid, material) {
  // 体素化完成时完整占据单元，随后才从单元中心逐渐缩小。
  const size = grid.cell_size;
  const geometry = new THREE.BoxGeometry(size, size, size);
  const mesh = new THREE.InstancedMesh(geometry, material, cells.length);
  const matrix = new THREE.Matrix4();
  const center = new THREE.Vector3();
  const color = new THREE.Color();
  cells.forEach((cell, index) => {
    center.set(...cell.map((coord, axis) => grid.lower[axis] + (coord + 0.5) * grid.cell_size));
    matrix.makeTranslation(center.x, center.y, center.z);
    mesh.setMatrixAt(index, matrix);
    const influence = 1 - THREE.MathUtils.smoothstep(center.distanceTo(focus), mapperFocus.innerRadius, mapperFocus.outerRadius);
    color.copy(darkMetal).lerp(brightMetal, influence);
    mesh.setColorAt(index, color);
  });
  mesh.instanceMatrix.needsUpdate = true;
  mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();
  return mesh;
}

function makeVoxelSet(cells, selected, focus, grid, tint, voxelize, shrink, dim) {
  // 选中的点始终不透明；未选中点使用独立实例网格，才能连续降低不透明度。
  const selectedMaterial = createVoxelMaterial(voxelize, shrink, grid, tint, focus, { value: 0 }, false);
  const contextMaterial = createVoxelMaterial(voxelize, shrink, grid, tint, focus, dim, true);
  const group = new THREE.Group();
  group.add(
    makeVoxelMesh(cells.filter((cell) => selected.has(cell.join(","))), focus, grid, selectedMaterial),
    makeVoxelMesh(cells.filter((cell) => !selected.has(cell.join(","))), focus, grid, contextMaterial),
  );
  group.visible = false;
  return { group, contextMaterial };
}

function makeVoxelSkeleton(cells, grid, color, focus, selected) {
  // 只连接局部六邻接占用体素的中心，不绘制各个立方体的十二条边。
  const localCenters = new Map();
  const selectedPositions = [];
  const contextPositions = [];
  const center = new THREE.Vector3();
  for (const cell of cells) {
    center.set(...cell.map((coord, axis) => grid.lower[axis] + (coord + 0.5) * grid.cell_size));
    if (center.distanceTo(focus) > mapperFocus.outerRadius + grid.cell_size / 2) continue;
    localCenters.set(cell.join(","), { cell, position: center.toArray() });
  }
  for (const { cell, position } of localCenters.values()) {
    for (let axis = 0; axis < 3; axis++) {
      const neighbor = [...cell];
      neighbor[axis]++;
      const other = localCenters.get(neighbor.join(","));
      if (!other) continue;
      const positions = selected.has(cell.join(",")) && selected.has(neighbor.join(",")) ? selectedPositions : contextPositions;
      positions.push(...position, ...other.position);
    }
  }
  function makeLines(positions) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0, depthWrite: false });
    return new THREE.LineSegments(geometry, material);
  }
  const selectedLines = makeLines(selectedPositions);
  const contextLines = makeLines(contextPositions);
  const group = new THREE.Group();
  group.add(contextLines, selectedLines);
  group.visible = false;
  return { group, selectedLines, contextLines };
}

function makeNeighborhoodOutline(center, radius, right, up, viewDirection) {
  // 圆形轮廓及一条倾斜大圆都使用真实三维半径，而不是屏幕像素半径。
  const segments = 128;
  const color = 0x75c3ff;
  function makeCircle(a, b, opacity) {
    const positions = [];
    for (let i = 0; i <= segments; i++) {
      const angle = 2 * Math.PI * i / segments;
      const point = center.clone().addScaledVector(a, radius * Math.cos(angle))
        .addScaledVector(b, radius * Math.sin(angle));
      positions.push(...point.toArray());
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const material = new THREE.LineDashedMaterial({
      color, dashSize: 0.009, gapSize: 0.006, transparent: true, opacity: 0, depthWrite: false,
    });
    const line = new THREE.Line(geometry, material);
    line.computeLineDistances();
    return { line, opacity, segments };
  }
  const inclined = up.clone().multiplyScalar(0.52).addScaledVector(viewDirection, Math.sqrt(1 - 0.52 ** 2));
  const rings = [makeCircle(right, up, 0.95), makeCircle(right, inclined, 0.28)];
  const group = new THREE.Group();
  group.add(...rings.map(({ line }) => line));
  group.visible = false;
  return { group, rings };
}

export async function createMapperIntro(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.toneMappingExposure = 1;
  renderer.autoClear = false;
  const [geometry, studioTexture, grid] = await Promise.all([
    loadSpinningTop(),
    new EXRLoader().setDataType(THREE.HalfFloatType).loadAsync(studioImage.href),
    loadVoxelGrid(),
  ]);
  studioTexture.mapping = THREE.EquirectangularReflectionMapping;
  function makeEnvironment() {
    const generator = new THREE.PMREMGenerator(renderer);
    const environment = generator.fromEquirectangular(studioTexture);
    generator.dispose();
    return environment;
  }
  let environment = makeEnvironment();
  const scene = new THREE.Scene();
  scene.environment = environment.texture;
  scene.environmentRotation.x = Math.PI / 2;
  scene.environmentIntensity = 1.15;
  const fill = new THREE.DirectionalLight(0xffffff, 0.65);
  fill.position.set(3, -4, 5);
  scene.add(fill, new THREE.HemisphereLight(0xffffff, 0x687180, 0.2));
  const progress = { value: 0 };
  const voxelize = { value: 0 };
  const shrink = { value: 0 };
  const dim = { value: 0 };
  const object = new THREE.Mesh(geometry, createMetalMaterial(progress, voxelize, grid));
  const rotatedObject = new THREE.Mesh(geometry, object.material);
  const sourceGroup = new THREE.Group();
  const rotatedGroup = new THREE.Group();
  const focus = new THREE.Vector3(...mapperFocus.position);
  const rotatedFocus = focus.clone().applyAxisAngle(new THREE.Vector3(0, 0, 1), 2 * Math.PI / 3);
  const neighborhood = selectMapperNeighborhood(grid);
  const sourceVoxels = makeVoxelSet(grid.source, neighborhood.sourceKeys, focus, grid, sourceColor, voxelize, shrink, dim);
  const rotatedVoxels = makeVoxelSet(grid.rotated, neighborhood.targetKeys, rotatedFocus, grid, rotatedColor, voxelize, shrink, dim);
  const sourceSkeleton = makeVoxelSkeleton(grid.source, grid, sourceColor, focus, neighborhood.sourceKeys);
  const rotatedSkeleton = makeVoxelSkeleton(grid.rotated, grid, rotatedColor, rotatedFocus, neighborhood.targetKeys);
  const frameGeometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(grid.side, grid.side, grid.side));
  const frameMaterial = new THREE.LineBasicMaterial({ color: 0xd4d8df, transparent: true, opacity: 0, depthWrite: false });
  const sourceFrame = new THREE.LineSegments(frameGeometry, frameMaterial);
  const rotatedFrame = new THREE.LineSegments(frameGeometry, frameMaterial);
  // 体素化阶段的右侧坐标系仍轴对齐，之后整个网格作为整体逆向旋转。
  const sourceGrid = new THREE.Group();
  const rotatedGrid = new THREE.Group();
  sourceGrid.add(sourceVoxels.group, sourceFrame, sourceSkeleton.group);
  rotatedGrid.add(rotatedVoxels.group, rotatedFrame, rotatedSkeleton.group);
  sourceGroup.add(object, sourceGrid);
  rotatedGroup.add(rotatedObject, rotatedGrid);
  rotatedGroup.visible = false;
  scene.add(sourceGroup, rotatedGroup);

  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20);
  const direction = new THREE.Vector3(3, -4, 3.8).normalize();
  const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 0, 1), direction).normalize();
  const up = new THREE.Vector3().crossVectors(direction, right).normalize();
  const radiusOutline = makeNeighborhoodOutline(neighborhood.targetPosition, neighborhood.radius, right, up, direction);
  sourceGroup.add(radiusOutline.group);
  // 根据该姿态下实际网格的投影边界居中，不移动或重缩放网格本身。
  const framingCenter = right.clone().multiplyScalar(0.1107504).addScaledVector(up, 0.01454196);
  const focusProjection = right.clone().multiplyScalar(focus.dot(right)).addScaledVector(up, focus.dot(up));
  camera.position.copy(framingCenter).addScaledVector(direction, 5);
  camera.up.set(0, 0, 1);
  camera.lookAt(framingCenter);

  const verticalSpan = 2.3;
  let active = false;
  let timeline = 0;
  let requestedTime = 0;
  let lastFrame = null;
  let frame = null;

  function invalidate() {
    if (active && frame === null) frame = requestAnimationFrame(draw);
  }

  function draw(now) {
    frame = null;
    if (!active || renderer.getContext().isContextLost()) return;
    if (lastFrame !== null) {
      const delta = now - lastFrame;
      const remaining = requestedTime - timeline;
      timeline += Math.sign(remaining) * Math.min(Math.abs(remaining), delta);
    }
    lastFrame = now;
    progress.value = ease(timeline / highlightMilliseconds);
    const split = ease((timeline - highlightMilliseconds) / splitMilliseconds);
    const rotation = ease((timeline - splitEnd) / rotationMilliseconds) * 2 * Math.PI / 3;
    const voxelTime = Math.max(0, timeline - rotationEnd);
    voxelize.value = THREE.MathUtils.clamp(voxelTime / voxelizeMilliseconds, 0, 1);
    const gridAppearance = ease(voxelTime / gridAppearMilliseconds);
    const align = ease((timeline - voxelizeEnd) / alignMilliseconds);
    const localZoom = ease((timeline - alignEnd) / localZoomMilliseconds);
    shrink.value = ease((timeline - localZoomEnd) / skeletonMilliseconds);
    dim.value = ease((timeline - skeletonEnd) / selectMilliseconds);
    const merge = ease((timeline - mergeStart) / mergeMilliseconds);
    const sphere = ease((timeline - mergeEnd) / sphereMilliseconds);
    const sharedView = timeline > selectEnd;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width < 2 || bounds.height < 2) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.max(1, Math.round(bounds.width * pixelRatio));
    const height = Math.max(1, Math.round(bounds.height * pixelRatio));
    if (canvas.width !== width || canvas.height !== height) renderer.setSize(width, height, false);
    const halfHeight = verticalSpan / 2;
    const halfWidth = halfHeight * width / height;
    camera.left = -halfWidth;
    camera.right = halfWidth;
    camera.top = halfHeight;
    camera.bottom = -halfHeight;
    camera.updateProjectionMatrix();

    // 第二个动作已经为任意 Z 轴旋转的外围立方体留出空间，体素化时取景完全不变。
    const cubeWidth = grid.side * (Math.SQRT2 * Math.hypot(right.x, right.y) + Math.abs(right.z));
    const cubeHeight = grid.side * (Math.SQRT2 * Math.hypot(up.x, up.y) + Math.abs(up.z));
    const cubeScale = Math.min(0.82, 0.84 * halfWidth / cubeWidth, 0.78 * verticalSpan / cubeHeight);
    // 同步放大到亮色邻域；左右使用同一物体坐标中的焦点和相同放大倍率。
    const localScale = Math.min(halfWidth, verticalSpan) / (2 * mapperFocus.outerRadius * 0.96);
    const scale = THREE.MathUtils.lerp(THREE.MathUtils.lerp(1, cubeScale, split), localScale, localZoom);
    const offset = halfWidth * 0.5 * split;
    const centerWeight = 1 - scale + scale * split;
    sourceGroup.scale.setScalar(scale);
    rotatedGroup.scale.setScalar(scale);
    sourceGroup.position.copy(framingCenter).multiplyScalar(centerWeight).addScaledVector(right, -offset)
      .addScaledVector(focusProjection, -scale * localZoom);
    rotatedGroup.position.copy(framingCenter).multiplyScalar(centerWeight).addScaledVector(right, offset * (1 - 2 * merge))
      .addScaledVector(focusProjection, -scale * localZoom);
    rotatedObject.rotation.z = rotation;
    rotatedGrid.rotation.z = -2 * Math.PI / 3 * align;
    object.visible = rotatedObject.visible = voxelize.value < 1;
    sourceVoxels.group.visible = rotatedVoxels.group.visible = voxelize.value > 0;
    for (const voxels of [sourceVoxels, rotatedVoxels]) {
      voxels.contextMaterial.opacity = 1 - 0.72 * dim.value;
      voxels.contextMaterial.depthWrite = dim.value === 0;
    }
    sourceFrame.visible = rotatedFrame.visible = gridAppearance > 0 && localZoom < 1;
    frameMaterial.opacity = 0.63 * gridAppearance * (1 - localZoom);
    for (const [skeleton, color] of [[sourceSkeleton, sourceColor], [rotatedSkeleton, rotatedColor]]) {
      skeleton.group.visible = shrink.value > 0;
      skeleton.selectedLines.material.opacity = 0.86 * shrink.value;
      skeleton.contextLines.material.opacity = 0.86 * shrink.value * (1 - 0.86 * dim.value);
      skeleton.contextLines.material.color.copy(color).lerp(contextGray, dim.value);
    }
    radiusOutline.group.visible = sphere > 0;
    for (const { line, opacity, segments } of radiusOutline.rings) {
      line.material.opacity = opacity * sphere;
      line.geometry.setDrawRange(0, Math.max(2, Math.round(segments * sphere) + 1));
    }

    // 两份网格独立进行深度测试，避免刚复制时重合的三角面产生闪烁。
    renderer.setScissorTest(false);
    renderer.clear(true, true, true);
    if (sharedView) {
      // 相机矩阵、投影、比例和第一帧位置都不改变；改为同一场景的一次真实深度渲染。
      sourceGroup.visible = rotatedGroup.visible = true;
      renderer.render(scene, camera);
      rotatedGroup.visible = false;
    } else {
      // 局部放大时两个独立视窗各自裁切，模型不会遮挡另一侧。
      const separatePanels = split >= 1;
      const leftWidth = Math.floor(width / 2);
      if (separatePanels) {
        renderer.setScissorTest(true);
        renderer.setScissor(0, 0, leftWidth, height);
      }
      sourceGroup.visible = true;
      rotatedGroup.visible = false;
      renderer.render(scene, camera);
      if (split > 0) {
        sourceGroup.visible = false;
        rotatedGroup.visible = true;
        if (separatePanels) renderer.setScissor(leftWidth, 0, width - leftWidth, height);
        renderer.clearDepth();
        renderer.render(scene, camera);
        sourceGroup.visible = true;
        rotatedGroup.visible = false;
      }
    }
    renderer.setScissorTest(false);
    if (timeline !== requestedTime) invalidate();
    else lastFrame = null;
  }

  const resizeObserver = new ResizeObserver(invalidate);
  resizeObserver.observe(canvas);
  canvas.addEventListener("webglcontextrestored", () => {
    environment.dispose();
    environment = makeEnvironment();
    scene.environment = environment.texture;
    invalidate();
  });

  return {
    start(stage) {
      if (!active) {
        active = true;
        progress.value = 0;
        timeline = 0;
        requestedTime = 0;
        lastFrame = null;
      }
      requestedTime = stageTimes[THREE.MathUtils.clamp(stage, 0, stageTimes.length - 1)];
      invalidate();
    },
    stop() {
      active = false;
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      lastFrame = null;
    },
  };
}
