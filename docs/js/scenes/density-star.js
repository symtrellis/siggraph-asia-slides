const corners = [
  [-94, 0.91], [-55, 0.43],
  [-21, 0.78], [12, 0.48],
  [51, 0.96], [92, 0.38],
  [129, 0.76], [158, 0.48],
  [205, 0.87], [235, 0.42],
  [266, 0.91],
];

function smoothstep(value) {
  const t = Math.max(0, Math.min(1, value));
  return t * t * (3 - 2 * t);
}

// 五条形状、长度和朝向略有差异的圆润星角。
function radiusAt(x, y) {
  let angle = Math.atan2(y, x) * 180 / Math.PI;
  if (angle < corners[0][0]) angle += 360;
  for (let i = 1; i < corners.length; i++) {
    const [endAngle, endRadius] = corners[i];
    if (angle > endAngle) continue;
    const [startAngle, startRadius] = corners[i - 1];
    const t = smoothstep((angle - startAngle) / (endAngle - startAngle));
    return startRadius + (endRadius - startRadius) * t;
  }
  return corners[0][1];
}

// 固定随机种子，使后续每次旋转使用完全相同的二维点。
export function makeStarPoints(pointCount = 64000) {
  let seed = 2026;
  function random() {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  }

  const points = [];
  const edgeWidth = 0.10;
  while (points.length < pointCount) {
    const x = 2.2 * random() - 1.1;
    const y = 2.2 * random() - 1.1;
    const edgeDistance = radiusAt(x, y) - Math.hypot(x, y);
    const probability = smoothstep((edgeDistance + edgeWidth / 2) / edgeWidth);
    if (random() < probability) points.push([x, y]);
  }
  return points;
}

// 每个点以一个小的高斯分布贡献密度，边缘从粉色自然过渡到透明。
export function drawDensityCloud(canvas, points) {
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  const density = new Float32Array(width * height);
  const scale = Math.min(width, height) * 0.44;
  const kernel = [];
  for (let dy = -3; dy <= 3; dy++) {
    for (let dx = -3; dx <= 3; dx++) {
      kernel.push([dx, dy, Math.exp(-(dx * dx + dy * dy) / (2 * 1.35 ** 2))]);
    }
  }

  for (const [x, y] of points) {
    const centerX = Math.round(width / 2 + scale * x);
    const centerY = Math.round(height / 2 + scale * y);
    for (const [dx, dy, weight] of kernel) {
      const px = centerX + dx;
      const py = centerY + dy;
      if (px < 0 || px >= width || py < 0 || py >= height) continue;
      density[py * width + px] += weight;
    }
  }

  const image = context.createImageData(width, height);
  for (let i = 0; i < density.length; i++) {
    const pixel = i * 4;
    image.data[pixel] = 246;
    image.data[pixel + 1] = 141;
    image.data[pixel + 2] = 164;
    image.data[pixel + 3] = 255 * (1 - Math.exp(-1.25 * density[i]));
  }
  context.putImageData(image, 0, 0);
}

const canvas = document.querySelector(".density-star");
drawDensityCloud(canvas, makeStarPoints());
