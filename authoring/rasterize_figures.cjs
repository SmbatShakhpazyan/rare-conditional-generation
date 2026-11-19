const fs = require('fs/promises');
const path = require('path');
const sharp = require('sharp');
(async () => {
  const dir = path.join(__dirname, '..', 'figures');
  const names = (await fs.readdir(dir)).filter(n=>n.endsWith('.svg'));
  const results = await Promise.allSettled(names.map(async n=> {
    await sharp(path.join(dir,n)).png().toFile(path.join(dir,n.replace(/\.svg$/,'.png')));
  }));
  for (const result of results) if(result.status!=='fulfilled') throw result.reason;
  process.stdout.write(JSON.stringify({rendered:names.length})+'\n');
})();
