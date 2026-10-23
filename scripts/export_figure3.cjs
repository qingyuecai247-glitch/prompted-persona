// Optional high-resolution PNG and PDF export of the canonical vector SVG.
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const { PDFDocument } = require('pdf-lib');
(async () => {
  const dir = process.argv[2] || path.resolve(__dirname, '../outputs/figures');
  const base = path.join(dir, 'figure3_compact');
  const source = fs.readFileSync(base + '.svg', 'utf8');
  const dimensions = source.match(/width="([\d.]+)mm" height="([\d.]+)mm" viewBox="0 0 ([\d.]+) ([\d.]+)"/);
  if (!dimensions) throw new Error('SVG physical size and viewBox are required.');
  const [, widthMM, heightMM, width, height] = dimensions.map(Number);
  const rasterSource = source.replace(/width="[\d.]+mm" height="[\d.]+mm"/, `width="${width}" height="${height}"`);
  const png = await sharp(Buffer.from(rasterSource), { density: 212 }).png().toBuffer();
  fs.writeFileSync(base + '.png', png);
  const pdf = await PDFDocument.create();
  const image = await pdf.embedPng(png);
  const page = pdf.addPage([widthMM / 25.4 * 72, heightMM / 25.4 * 72]);
  page.drawImage(image, { x: 0, y: 0, width: page.getWidth(), height: page.getHeight() });
  fs.writeFileSync(base + '.pdf', await pdf.save());
  console.log('Exported PNG and raster PDF; SVG remains the editable vector master.');
})().catch(error => { console.error(error.message); process.exitCode = 1; });
