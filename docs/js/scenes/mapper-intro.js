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

// 海报中的静态演示：三次旋转分别体素化，然后在同一网格中取占用平均。
export async function createVelocitySymmetrizationScene(canvas) {
  const [geometry, studioTexture, grid] = await Promise.all([
    loadSpinningTop(),
    new EXRLoader().setDataType(THREE.HalfFloatType).loadAsync(studioImage.href),
    loadVoxelGrid(),
  ]);
  if (!Array.isArray(grid.rotated_240) || grid.rotated_240.length === 0) {
    throw new Error("缺少 240° 姿态的共同坐标系体素占用数据");
  }

  const renderer = new THREE.WebGLRenderer({
    canvas, alpha: true, antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0xffffff, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.toneMappingExposure = 1;
  renderer.autoClear = false;

  studioTexture.mapping = THREE.EquirectangularReflectionMapping;
  function makeEnvironment() {
    const generator = new THREE.PMREMGenerator(renderer);
    const environment = generator.fromEquirectangular(studioTexture);
    generator.dispose();
    return environment;
  }
  let environment = makeEnvironment();
  const direction = new THREE.Vector3(3, -4, 3.8).normalize();
  const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 0, 1), direction).normalize();
  const up = new THREE.Vector3().crossVectors(direction, right).normalize();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 40);
  camera.position.copy(direction).multiplyScalar(12);
  camera.up.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  const metal = new THREE.MeshStandardMaterial({ color: 0x909090, metalness: 0.92, roughness: 0.28, side: THREE.DoubleSide });
  const voxelMaterial = new THREE.MeshStandardMaterial({ color: 0x777777, metalness: 0.08, roughness: 0.68 });
  const palette = [0xd75c58, 0xbc8a13, 0x388ac6];
  // 留出很窄的显示间隙，避免相邻透明立方体的重合面闪烁；真实网格索引不变。
  const cubeSide = grid.cell_size * 0.94;
  const cubeGeometry = new THREE.BoxGeometry(cubeSide, cubeSide, cubeSide);

  const scene = new THREE.Scene();
  scene.environment = environment.texture;
  scene.environmentRotation.x = Math.PI / 2;
  scene.environmentIntensity = 1.15;
  const fill = new THREE.DirectionalLight(0xffffff, 0.8);
  fill.position.set(3, -4, 5);
  scene.add(fill, new THREE.HemisphereLight(0xffffff, 0x737373, 0.35));

  function makeVoxelMesh(cells, material, opacities = null) {
    // 这个静态示意图只有几千个体素。直接合并每个立方体的顶点，避免多组实例绘制时偶尔缺失。
    const vertices = cubeGeometry.attributes.position;
    const normals = cubeGeometry.attributes.normal;
    const faces = cubeGeometry.index.array;
    const positions = new Float32Array(cells.length * vertices.count * 3);
    const normalData = new Float32Array(positions.length);
    const indices = new Uint32Array(cells.length * faces.length);
    const alpha = opacities ? new Float32Array(cells.length * vertices.count) : null;
    cells.forEach((cell, index) => {
      const center = cellCenter(cell, grid);
      const firstVertex = index * vertices.count;
      for (let v = 0; v < vertices.count; v++) {
        const offset = (firstVertex + v) * 3;
        positions[offset] = vertices.getX(v) + center.x;
        positions[offset + 1] = vertices.getY(v) + center.y;
        positions[offset + 2] = vertices.getZ(v) + center.z;
        normalData[offset] = normals.getX(v);
        normalData[offset + 1] = normals.getY(v);
        normalData[offset + 2] = normals.getZ(v);
        if (alpha) alpha[firstVertex + v] = opacities[index];
      }
      for (let f = 0; f < faces.length; f++) indices[index * faces.length + f] = firstVertex + faces[f];
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(normalData, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    if (alpha) geometry.setAttribute("cellOpacity", new THREE.BufferAttribute(alpha, 1));
    geometry.computeBoundingSphere();
    return new THREE.Mesh(geometry, material);
  }

  function makeRotationArrow(angle) {
    const radius = 0.58;
    const z = 0.83;
    // 两个箭头使用同一个起点，在选定的右上方参考方向基础上顺时针转 90°。
    const start = Math.atan2(right.y, right.x) + Math.PI + 4 * Math.PI / 3 - Math.PI / 2;
    const end = start + angle;
    const headLength = 0.16;
    const headRadius = 0.075;
    const material = new THREE.MeshBasicMaterial({ color: 0x30343a, toneMapped: false, depthTest: false, depthWrite: false });
    const group = new THREE.Group();
    // 弧线从同一个参考方向出发，箭头尖端严格落在绕 Z 轴转过 angle 的终点。
    const shaftEnd = end - 0.78 * headLength / radius;
    const arc = new THREE.Curve();
    arc.getPoint = (t, target = new THREE.Vector3()) => {
      const a = THREE.MathUtils.lerp(start, shaftEnd, t);
      return target.set(radius * Math.cos(a), radius * Math.sin(a), z);
    };
    const shaft = new THREE.Mesh(new THREE.TubeGeometry(arc, 64, 0.026, 8, false), material);
    const head = new THREE.Mesh(new THREE.ConeGeometry(headRadius, headLength, 16), material);
    const tangent = new THREE.Vector3(-Math.sin(end), Math.cos(end), 0);
    head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
    head.position.set(radius * Math.cos(end), radius * Math.sin(end), z).addScaledVector(tangent, -headLength / 2);
    shaft.renderOrder = head.renderOrder = 10;
    group.add(shaft, head);
    return group;
  }

  const poses = [grid.source, grid.rotated, grid.rotated_240];
  const meshPanels = poses.map((_, index) => {
    const object = new THREE.Mesh(geometry, metal);
    object.rotation.z = index * 2 * Math.PI / 3;
    const group = new THREE.Group();
    group.add(object);
    if (index > 0) group.add(makeRotationArrow(object.rotation.z));
    scene.add(group);
    return group;
  });
  const voxelPanels = poses.map((cells, index) => {
    const material = voxelMaterial.clone();
    material.color.setHex(palette[index]);
    const object = makeVoxelMesh(cells, material);
    scene.add(object);
    return object;
  });

  // 先按整数网格索引求和，再除以三。不同姿态并不各自重新放缩或平移。
  const occupancy = new Map();
  for (const cells of poses) {
    for (const cell of cells) {
      const key = cell.join(",");
      const value = occupancy.get(key);
      if (value) value.count += 1;
      else occupancy.set(key, { cell, count: 1 });
    }
  }
  const fullCells = [...occupancy.values()].filter(({ count }) => count === 3);
  const partialCells = [...occupancy.values()].filter(({ count }) => count < 3);
  // 透明体素按实际相机深度排序，一个体素只绘制一次，占用次数就是不透明度。
  partialCells.sort((a, b) => cellCenter(a.cell, grid).dot(direction) - cellCenter(b.cell, grid).dot(direction));
  const partialMaterial = voxelMaterial.clone();
  partialMaterial.transparent = true;
  partialMaterial.depthWrite = false;
  partialMaterial.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float cellOpacity; varying float vCellOpacity;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvCellOpacity = cellOpacity;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vCellOpacity;")
      .replace("#include <color_fragment>", "#include <color_fragment>\ndiffuseColor.a *= vCellOpacity;");
  };
  const average = new THREE.Group();
  average.add(makeVoxelMesh(fullCells.map(({ cell }) => cell), voxelMaterial));
  average.add(makeVoxelMesh(partialCells.map(({ cell }) => cell), partialMaterial, partialCells.map(({ count }) => count / 3)));
  scene.add(average);

  let frame = null;
  function render() {
    frame = null;
    if (renderer.getContext().isContextLost()) return;
    const size = canvas.getBoundingClientRect();
    if (size.width < 2 || size.height < 2) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.round(size.width * pixelRatio);
    const height = Math.round(size.height * pixelRatio);
    if (canvas.width !== width || canvas.height !== height) renderer.setSize(width, height, false);
    const originalAverageLeft = 3 * Math.floor(width * 0.60 / 3);
    const smallWidth = Math.floor(width * 0.66 / 3);
    const rowHeight = Math.floor(height / 2);
    const averageLeft = 3 * smallWidth;
    const smallSpan = Math.min(smallWidth, rowHeight);
    const worldPerPixel = 2.25 / smallSpan;
    camera.left = -width * worldPerPixel / 2;
    camera.right = width * worldPerPixel / 2;
    camera.top = height * worldPerPixel / 2;
    camera.bottom = -height * worldPerPixel / 2;
    camera.updateProjectionMatrix();

    // 在同一个正交相机下布局七个物体，而不是连续切换七个画布裁切区域。
    function place(object, x, y, panelWidth, panelHeight, fittedPixels = Math.min(panelWidth, panelHeight), centerOffset = 0) {
      object.scale.setScalar(fittedPixels / smallSpan);
      object.position.copy(right).multiplyScalar((x + panelWidth / 2 + centerOffset - width / 2) * worldPerPixel)
        .addScaledVector(up, (y + panelHeight / 2 - height / 2) * worldPerPixel);
    }
    for (let index = 0; index < 3; index++) {
      place(meshPanels[index], index * smallWidth, rowHeight, smallWidth, height - rowHeight, 1.15 * Math.min(smallWidth, height - rowHeight));
      place(voxelPanels[index], index * smallWidth, 0, smallWidth, rowHeight, 1.15 * Math.min(smallWidth, rowHeight));
    }
    // 只增大三个旋转姿态；平均结果使用改动之前的投影大小和中心位置。
    place(average, averageLeft, 0, width - averageLeft, height,
      Math.min(width - originalAverageLeft, height), (originalAverageLeft - averageLeft) / 2);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, width, height);
    renderer.clear(true, true, true);
    renderer.render(scene, camera);
  }

  function invalidate() {
    if (frame === null) frame = requestAnimationFrame(render);
  }
  const resizeObserver = new ResizeObserver(invalidate);
  canvas.addEventListener("webglcontextrestored", () => {
    environment.dispose();
    environment = makeEnvironment();
    scene.environment = environment.texture;
    invalidate();
  });
  return {
    start() {
      resizeObserver.observe(canvas);
      invalidate();
    },
    stop() {
      resizeObserver.disconnect();
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
    },
  };
}

// 海报的静态 Mapper 示意图。物体始终处于原姿态，目标坐标架相对原坐标架转过 -120°。
export async function createPosterMapperScene(canvas, guides) {
  const [geometry, studioTexture, grid] = await Promise.all([
    loadSpinningTop(),
    new EXRLoader().setDataType(THREE.HalfFloatType).loadAsync(studioImage.href),
    loadVoxelGrid(),
  ]);
  const renderer = new THREE.WebGLRenderer({
    canvas, alpha: true, antialias: true, preserveDrawingBuffer: true, powerPreference: "high-performance",
  });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0xffffff, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.toneMappingExposure = 1;
  renderer.autoClear = false;

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
  const fill = new THREE.DirectionalLight(0xffffff, 0.75);
  fill.position.set(3, -4, 5);
  scene.add(fill, new THREE.HemisphereLight(0xffffff, 0x747474, 0.35));

  const azimuth = THREE.MathUtils.degToRad(-105);
  const elevation = THREE.MathUtils.degToRad(23);
  const direction = new THREE.Vector3(Math.cos(azimuth) * Math.cos(elevation), Math.sin(azimuth) * Math.cos(elevation), Math.sin(elevation));
  const axis = new THREE.Vector3(0, 0, 1);
  const right = new THREE.Vector3().crossVectors(axis, direction).normalize();
  const up = new THREE.Vector3().crossVectors(direction, right).normalize();
  const fieldOfView = 12;
  const halfFovTangent = Math.tan(THREE.MathUtils.degToRad(fieldOfView / 2));
  // 每个画格都使用真正的透视相机，相机方向和视野角完全相同。
  function makeCamera(center, distance, aspect) {
    const camera = new THREE.PerspectiveCamera(fieldOfView, aspect, 0.015, 40);
    camera.position.copy(center).addScaledVector(direction, distance);
    camera.up.copy(axis);
    camera.lookAt(center);
    camera.updateMatrixWorld();
    return camera;
  }

  const angle = 2 * Math.PI / 3;
  const focus = new THREE.Vector3(...mapperFocus.position);
  const rotatedFocus = focus.clone().applyAxisAngle(axis, angle);
  const neighborhood = selectMapperNeighborhood(grid);
  const contextLimit = mapperFocus.outerRadius + grid.cell_size / 2;
  const sourceLocal = grid.source.filter((cell) => cellCenter(cell, grid).distanceTo(focus) <= contextLimit);
  const targetLocal = grid.rotated.filter((cell) => cellCenter(cell, grid).distanceTo(rotatedFocus) <= contextLimit);
  const dark = new THREE.Color(0x62676d);
  const white = new THREE.Color(0xfafafa);
  const faint = new THREE.Color(0xb1b5b9);
  const lineColor = new THREE.Color(0x747a81);
  const sourceTint = sourceColor.clone();
  const targetTint = new THREE.Color(0x388ac6);
  const metal = createMetalMaterial({ value: 1 }, { value: 0 }, grid);
  const voxelMaterial = new THREE.MeshStandardMaterial({
    color: 0xffffff, vertexColors: true, metalness: 0.04, roughness: 0.65,
  });
  const localVoxelMaterial = voxelMaterial.clone();
  localVoxelMaterial.envMapIntensity = 0.4;
  localVoxelMaterial.roughness = 0.48;
  const dimVoxelMaterial = localVoxelMaterial.clone();
  dimVoxelMaterial.transparent = true;
  dimVoxelMaterial.depthWrite = false;
  dimVoxelMaterial.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float posterOpacity; varying float vPosterOpacity;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvPosterOpacity = posterOpacity;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vPosterOpacity;")
      .replace("#include <color_fragment>", "#include <color_fragment>\ndiffuseColor.a *= vPosterOpacity;");
  };
  const frameGeometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(grid.side, grid.side, grid.side));
  const frameMaterial = new THREE.LineBasicMaterial({ color: lineColor, transparent: true, opacity: 0.75, depthWrite: false });

  function makeFrame(rotation) {
    const frame = new THREE.LineSegments(frameGeometry, frameMaterial);
    frame.rotation.z = rotation;
    return frame;
  }

  function makeVoxelGeometry(cells, sideFraction, colorForCell, opacityForCell = null) {
    const box = new THREE.BoxGeometry(...Array(3).fill(grid.cell_size * sideFraction));
    const position = box.attributes.position;
    const normal = box.attributes.normal;
    const faces = box.index.array;
    const positions = new Float32Array(cells.length * position.count * 3);
    const normals = new Float32Array(positions.length);
    const colors = new Float32Array(positions.length);
    const opacities = opacityForCell ? new Float32Array(cells.length * position.count) : null;
    const indices = new Uint32Array(cells.length * faces.length);
    cells.forEach((cell, index) => {
      const center = cellCenter(cell, grid);
      const color = colorForCell(cell, center);
      const opacity = opacityForCell ? opacityForCell(cell, center) : 1;
      const first = index * position.count;
      for (let v = 0; v < position.count; v++) {
        const offset = (first + v) * 3;
        positions.set([position.getX(v) + center.x, position.getY(v) + center.y, position.getZ(v) + center.z], offset);
        normals.set([normal.getX(v), normal.getY(v), normal.getZ(v)], offset);
        colors.set([color.r, color.g, color.b], offset);
        if (opacities) opacities[first + v] = opacity;
      }
      for (let f = 0; f < faces.length; f++) indices[index * faces.length + f] = first + faces[f];
    });
    box.dispose();
    const result = new THREE.BufferGeometry();
    result.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    result.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    result.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    if (opacities) result.setAttribute("posterOpacity", new THREE.BufferAttribute(opacities, 1));
    result.setIndex(new THREE.BufferAttribute(indices, 1));
    result.computeBoundingSphere();
    return result;
  }

  function makeVoxels(cells, sideFraction, colorForCell, material = voxelMaterial, opacityForCell = null) {
    return new THREE.Mesh(makeVoxelGeometry(cells, sideFraction, colorForCell, opacityForCell), material);
  }

  function makeModelPanel(rotation) {
    const group = new THREE.Group();
    // 上下模型姿态都为零；只有外围坐标架转动。
    group.add(new THREE.Mesh(geometry, metal), makeFrame(rotation));
    scene.add(group);
    return group;
  }

  function makeGlobalVoxelPanel(cells, localFocus, rotation) {
    const group = new THREE.Group();
    // 第二列保持完整的体素边长；只显示原缓存中的三角形相交表面体素。
    const voxels = makeVoxels(cells, 1.0, (_cell, center) => {
      const influence = 1 - THREE.MathUtils.smoothstep(center.distanceTo(localFocus), mapperFocus.innerRadius, mapperFocus.outerRadius);
      return dark.clone().lerp(white, influence);
    });
    voxels.rotation.z = rotation;
    group.add(voxels);
    scene.add(group);
    return group;
  }

  function makeFullSkeleton(cells, activeKeys, tint) {
    const cellKeys = new Set(cells.map((cell) => cell.join(",")));
    const activePositions = [];
    const contextPositions = [];
    for (const cell of cells) {
      const key = cell.join(",");
      const center = cellCenter(cell, grid).toArray();
      for (let dimension = 0; dimension < 3; dimension++) {
        const adjacent = [...cell];
        adjacent[dimension]++;
        const neighborKey = adjacent.join(",");
        if (!cellKeys.has(neighborKey)) continue;
        const selected = activeKeys.has(key) && activeKeys.has(neighborKey);
        const positions = selected ? activePositions : contextPositions;
        positions.push(...center, ...cellCenter(adjacent, grid).toArray());
      }
    }
    function lines(positions, color, opacity) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
      return new THREE.LineSegments(geometry, material);
    }
    const group = new THREE.Group();
    group.add(lines(activePositions, tint, 0.75), lines(contextPositions, faint, 0.16));
    return group;
  }

  function makeLocalSchematic(cells, localFocus, rotation, tint, activeKeys) {
    const group = new THREE.Group();
    const active = cells.filter((cell) => activeKeys.has(cell.join(",")));
    // 其余所有表面体素仍然存在；只改变颜色和透明度，不再裁掉局部区域外的网格。
    const context = cells.filter((cell) => !activeKeys.has(cell.join(","))).map((cell) => ({
      cell, depth: cellCenter(cell, grid).applyAxisAngle(axis, rotation).dot(direction),
    })).sort((a, b) => a.depth - b.depth).map(({ cell }) => cell);
    if (active.length) group.add(makeVoxels(active, 0.36, () => tint, localVoxelMaterial));
    if (context.length) group.add(makeVoxels(context, 0.27, () => faint, dimVoxelMaterial, (_cell, center) =>
      0.05 + 0.24 * (1 - THREE.MathUtils.smoothstep(center.distanceTo(localFocus), contextLimit, contextLimit + 0.35))));
    group.add(makeFullSkeleton(cells, activeKeys, tint));
    group.rotation.z = rotation;
    return group;
  }

  const modelPanels = [makeModelPanel(0), makeModelPanel(-angle)];
  const voxelPanels = [
    makeGlobalVoxelPanel(grid.source, focus, 0),
    makeGlobalVoxelPanel(grid.rotated, rotatedFocus, -angle),
  ];
  const localPanels = [
    makeLocalSchematic(grid.source, focus, 0, sourceTint, new Set(sourceLocal.map((cell) => cell.join(",")))),
    makeLocalSchematic(grid.rotated, rotatedFocus, -angle, targetTint, new Set(targetLocal.map((cell) => cell.join(",")))),
  ];
  scene.add(...localPanels);
  const merged = new THREE.Group();
  merged.add(
    makeLocalSchematic(grid.source, focus, 0, sourceTint, neighborhood.sourceKeys),
    makeLocalSchematic(grid.rotated, rotatedFocus, -angle, targetTint, neighborhood.targetKeys),
  );
  const ball = makeNeighborhoodOutline(neighborhood.targetPosition, neighborhood.radius, right, up, direction);
  ball.group.visible = true;
  for (const ring of ball.rings) {
    ring.line.material.color.setHex(0x388ac6);
    ring.line.material.opacity = ring.opacity;
    ring.line.material.depthTest = false;
    ring.line.renderOrder = 20;
  }
  merged.add(ball.group);
  scene.add(merged);

  // 根据两个真实坐标架确定统一取景，第一列和第二列共用同一相机距离。
  const frameCorners = [];
  for (const rotation of [0, -angle]) {
    for (const x of [-grid.side / 2, grid.side / 2]) {
      for (const y of [-grid.side / 2, grid.side / 2]) {
        for (const z of [-grid.side / 2, grid.side / 2]) frameCorners.push(new THREE.Vector3(x, y, z).applyAxisAngle(axis, rotation));
      }
    }
  }
  function fitDistance(points, aspect, coverage) {
    let low = 1.6;
    let high = 32;
    for (let step = 0; step < 26; step++) {
      const distance = (low + high) / 2;
      const camera = makeCamera(new THREE.Vector3(), distance, aspect);
      const fits = points.every((point) => {
        const projected = point.clone().project(camera);
        return Math.abs(projected.x) < coverage && Math.abs(projected.y) < coverage && projected.z < 1;
      });
      if (fits) high = distance;
      else low = distance;
    }
    return high;
  }

  function updateZoomGuides(width, height, zoomPairs, mergedPanel) {
    const namespace = "http://www.w3.org/2000/svg";
    guides.setAttribute("viewBox", `0 0 ${width} ${height}`);
    function svgElement(tag, attributes) {
      const element = document.createElementNS(namespace, tag);
      for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
      return element;
    }
    const definitions = svgElement("defs", {});
    const elements = [definitions];
    // 用白底渐变遮罩消除局部视窗的直线裁切边缘，实际网格仍完整保留。
    for (const [index, panel] of [...zoomPairs.map((pair) => pair[1]), mergedPanel].entries()) {
      const side = Math.min(panel.width, panel.height, panel.fadeSide ?? Infinity);
      const innerSide = 0.84 * side;
      const square = {
        x: panel.x + (panel.width - innerSide) / 2,
        y: panel.y + (panel.height - innerSide) / 2,
        width: innerSide,
        height: innerSide,
        rx: 0.14 * side,
      };
      const filterId = `poster-mapper-edge-filter-${index}`;
      const maskId = `poster-mapper-edge-mask-${index}`;
      const panelBounds = { x: panel.x, y: panel.y, width: panel.width, height: panel.height };
      const filter = svgElement("filter", { id: filterId, filterUnits: "userSpaceOnUse", ...panelBounds });
      filter.append(svgElement("feGaussianBlur", { stdDeviation: 0.04 * side }));
      const mask = svgElement("mask", { id: maskId, maskUnits: "userSpaceOnUse", maskContentUnits: "userSpaceOnUse", "mask-type": "luminance", ...panelBounds });
      mask.append(
        svgElement("rect", { ...panelBounds, fill: "white" }),
        svgElement("rect", { ...square, fill: "black", filter: `url(#${filterId})` }),
      );
      definitions.append(filter, mask);
      elements.push(svgElement("rect", { ...panelBounds, fill: "white", mask: `url(#${maskId})` }));
      // 渐变必须在裁切边界之前完全结束；实白外沿消除浏览器亚像素合成留下的细线。
      const guard = Math.max(3, 0.025 * side);
      const bleed = 1.5;
      const outerX = panel.x - bleed;
      const outerY = panel.y - bleed;
      const outerWidth = panel.width + 2 * bleed;
      const outerHeight = panel.height + 2 * bleed;
      for (const edge of [
        { x: outerX, y: outerY, width: guard + bleed, height: outerHeight },
        { x: panel.x + panel.width - guard, y: outerY, width: guard + bleed, height: outerHeight },
        { x: outerX, y: outerY, width: outerWidth, height: guard + bleed },
        { x: outerX, y: panel.y + panel.height - guard, width: outerWidth, height: guard + bleed },
      ]) elements.push(svgElement("rect", { ...edge, fill: "white" }));
    }
    function project(point, panel) {
      const projected = point.clone().project(panel.camera);
      return new THREE.Vector2(panel.x + (projected.x + 1) * panel.width / 2, panel.y + (1 - projected.y) * panel.height / 2);
    }
    for (const [row, [overview, zoom]] of zoomPairs.entries()) {
      // 两端的圆都代表同一个实际三维区域，半径由各自的透视投影计算。
      const edge = focus.clone().addScaledVector(right, mapperFocus.outerRadius);
      const smallCenter = project(focus, overview);
      const largeCenter = project(focus, zoom);
      const smallRadius = smallCenter.distanceTo(project(edge, overview));
      const largeRadius = largeCenter.distanceTo(project(edge, zoom));
      const delta = largeCenter.clone().sub(smallCenter);
      const separation = delta.length();
      const along = delta.multiplyScalar(1 / separation);
      const perpendicular = new THREE.Vector2(-along.y, along.x);
      const projection = THREE.MathUtils.clamp((smallRadius - largeRadius) / separation, -0.99, 0.99);
      const tangentWeight = Math.sqrt(1 - projection * projection);
      const tangents = [1, -1].map((sign) => {
        const normal = along.clone().multiplyScalar(projection).addScaledVector(perpendicular, sign * tangentWeight);
        return [smallCenter.clone().addScaledVector(normal, smallRadius), largeCenter.clone().addScaledVector(normal, largeRadius)];
      });
      const gradientId = `poster-mapper-zoom-cone-${row}`;
      const fadeEndX = Math.max(smallCenter.x + 1, zoom.x - 0.02 * zoom.width);
      const fadeFraction = (fadeEndX - smallCenter.x) / (largeCenter.x - smallCenter.x);
      const gradient = svgElement("linearGradient", {
        id: gradientId, gradientUnits: "userSpaceOnUse",
        x1: smallCenter.x, y1: smallCenter.y, x2: fadeEndX,
        y2: smallCenter.y + (largeCenter.y - smallCenter.y) * fadeFraction,
      });
      gradient.append(
        svgElement("stop", { offset: "0%", "stop-color": "#aeb5bc", "stop-opacity": "0.32" }),
        svgElement("stop", { offset: "55%", "stop-color": "#aeb5bc", "stop-opacity": "0.20" }),
        svgElement("stop", { offset: "100%", "stop-color": "#aeb5bc", "stop-opacity": "0" }),
      );
      definitions.append(gradient);
      const [[smallA, largeA], [smallB, largeB]] = tangents;
      const cone = svgElement("path", {
        d: `M ${smallA.x} ${smallA.y} L ${largeA.x} ${largeA.y} L ${largeB.x} ${largeB.y} L ${smallB.x} ${smallB.y} Z`,
        fill: `url(#${gradientId})`,
      });
      const circle = document.createElementNS(namespace, "circle");
      circle.setAttribute("cx", smallCenter.x);
      circle.setAttribute("cy", smallCenter.y);
      circle.setAttribute("r", smallRadius);
      circle.setAttribute("fill", "none");
      circle.setAttribute("stroke", "#30343a");
      circle.setAttribute("stroke-width", "1.8");
      const circleBorder = circle.cloneNode();
      circleBorder.setAttribute("stroke", "#ffffff");
      circleBorder.setAttribute("stroke-width", "4.2");
      elements.push(cone, circleBorder, circle);
    }
    guides.replaceChildren(...elements);
  }

  let frame = null;
  function render() {
    frame = null;
    if (renderer.getContext().isContextLost()) return;
    const bounds = canvas.getBoundingClientRect();
    if (bounds.width < 2 || bounds.height < 2) return;
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.round(bounds.width * pixelRatio);
    const height = Math.round(bounds.height * pixelRatio);
    if (canvas.width !== width || canvas.height !== height) renderer.setSize(width, height, false);
    const columnWidths = [0.22, 0.19, 0.24].map((fraction) => Math.floor(width * fraction));
    columnWidths.push(width - columnWidths.reduce((sum, value) => sum + value, 0));
    const columnStarts = [0, columnWidths[0], columnWidths[0] + columnWidths[1], columnWidths[0] + columnWidths[1] + columnWidths[2]];
    const rowHeight = Math.floor(height / 2);
    const aspect = columnWidths[0] / rowHeight;
    const modelDistance = fitDistance(frameCorners, aspect, 0.94);
    const voxelDistance = modelDistance;
    const zoomDistance = contextLimit * rowHeight / (halfFovTangent * 0.88 * Math.min(columnWidths[2], rowHeight));
    const mergedWidth = columnWidths[3];
    const mergedFocusWidth = Math.min(mergedWidth, 1.08 * columnWidths[2]);
    const mergedDrawWidth = Math.min(mergedWidth, Math.round(1.16 * columnWidths[2]));
    const mergedPanelHeight = Math.floor(height * 0.60);
    const mergedDistance = contextLimit * mergedPanelHeight / (halfFovTangent * 0.92 * Math.min(mergedFocusWidth, mergedPanelHeight));

    const allPanels = [...modelPanels, ...voxelPanels, ...localPanels, merged];
    allPanels.forEach((object) => { object.visible = false; });
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, width, height);
    renderer.clear(true, true, true);
    renderer.setScissorTest(true);
    function draw(object, column, row, center, distance, panelWidth = columnWidths[column], panelHeight = rowHeight) {
      const x = columnStarts[column];
      const y = row === null ? 0 : row * rowHeight;
      const camera = makeCamera(center, distance, panelWidth / panelHeight);
      const bottom = height - y - panelHeight;
      renderer.setViewport(x, bottom, panelWidth, panelHeight);
      renderer.setScissor(x, bottom, panelWidth, panelHeight);
      object.visible = true;
      renderer.render(scene, camera);
      object.visible = false;
      return { camera, x, y, width: panelWidth, height: panelHeight };
    }
    const zoomPairs = [];
    for (let row = 0; row < 2; row++) {
      draw(modelPanels[row], 0, row, new THREE.Vector3(), modelDistance);
      const overview = draw(voxelPanels[row], 1, row, new THREE.Vector3(), voxelDistance);
      const zoom = draw(localPanels[row], 2, row, focus, zoomDistance);
      zoomPairs.push([overview, zoom]);
    }
    const mergedPanel = draw(merged, 3, null, focus, mergedDistance, mergedDrawWidth, mergedPanelHeight);
    renderer.setScissorTest(false);
    updateZoomGuides(width, height, zoomPairs, mergedPanel);
  }
  function invalidate() {
    if (frame === null) frame = requestAnimationFrame(render);
  }
  const observer = new ResizeObserver(invalidate);
  canvas.addEventListener("webglcontextrestored", () => {
    environment.dispose();
    environment = makeEnvironment();
    scene.environment = environment.texture;
    invalidate();
  });
  return {
    start() { observer.observe(canvas); invalidate(); },
    stop() {
      observer.disconnect();
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
    },
  };
}
