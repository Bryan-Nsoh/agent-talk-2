#!/usr/bin/env node
const path = require('path');
const fs = require('fs');
const PptxGenJS = require('pptxgenjs');
const html2pptx = require(path.resolve(__dirname, '../../../slide-maker/scripts/html2pptx'));

async function main() {
  const pptx = new PptxGenJS();
  pptx.layout = 'LAYOUT_16x9';

  const slidesDir = path.resolve(__dirname, '../../../slide-maker/workspace/html-slides/agent-talk');
  const htmlFiles = fs.readdirSync(slidesDir)
    .filter(f => f.endsWith('.html'))
    .sort();

  for (const file of htmlFiles) {
    const full = path.join(slidesDir, file);
    await html2pptx(full, pptx);
    console.log('Added slide from', file);
  }

  const outPath = path.resolve(__dirname, '../../../slide-maker/workspace/presentations/agent-talk.pptx');
  await pptx.writeFile(outPath);
  console.log('Wrote', outPath);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
