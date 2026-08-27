const fs = require('fs')
const path = require('path')

async function prepareModels() {
  const modelsDir = path.join(__dirname, '..', 'public', 'models')
  const defaultModel = path.join(modelsDir, 'super-resolution-10.onnx')

  if (!fs.existsSync(defaultModel)) {
    console.error('Default model super-resolution-10.onnx missing!')
    return
  }

  const modelFiles = ['edsr.onnx', 'fsrcnn.onnx', 'real-esrgan.onnx']
  for (const name of modelFiles) {
    const target = path.join(modelsDir, name)
    let needsCopy = false
    if (!fs.existsSync(target)) {
      needsCopy = true
    } else {
      const stat = fs.statSync(target)
      if (stat.size === 0) needsCopy = true
    }

    if (needsCopy) {
      fs.copyFileSync(defaultModel, target)
      console.log(`Prepared model placeholder ${name} from super-resolution-10.onnx`)
    }
  }
}

prepareModels().catch(err => {
  console.error('Error preparing models:', err)
})
