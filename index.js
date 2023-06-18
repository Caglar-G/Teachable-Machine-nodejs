import * as tf from '@tensorflow/tfjs-node'
import * as fs from 'fs';
console.log("start")


//load model ".savedmodel"
const model = await tf.node.loadSavedModel("./model/model.savedmodel");

/**
 * load image to guess
 * image must be 244x244
 */
const imageBuffer = fs.readFileSync("./localTest/test6.jpg");

//image decodeJpeg
const tfimage = tf.node.decodeJpeg(imageBuffer, 3);

/**
 * Adjustments are made for the teachable machine model
 * We convert it because Tensor must 4D and float
 */
const inputTensor = tfimage.expandDims(0).toFloat().div(tf.scalar(127.5)).sub(tf.scalar(1));

// Call predict
const prediction = await model.predict(inputTensor)

// Analyzing results
const classes = await getTopKClasses(["empty","opel","diffrentCar"]
, prediction, 3);

console.log('Classes:', classes);

const topResult = classes[0].className;
console.log('Prediction:', topResult);




/**
 * 
 * @param {Array<string>} labels - The list array for labels
 * @param {Tensor} logits - tensor returned for result
 * @param {int} topK 
 * @returns 
 */
async function getTopKClasses(labels, logits, topK = 3) {
    const values = await logits.dataSync();
    topK = Math.min(topK, values.length);
  
    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({ value: values[i], index: i });
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }
  
    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesAndProbs.push({
        className: labels[topkIndices[i]],
        probability: topkValues[i],
      });
    }
    return topClassesAndProbs;
}