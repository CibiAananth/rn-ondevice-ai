import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import {bundleResourceIO, decodeJpeg} from '@tensorflow/tfjs-react-native';

import {Base64Binary} from '../util';

const modelJson = require('../converted/model.json');

const weights = [];
weights.push(require('../converted/1.bin'));

export const BITMAP_DIMENSION = 224;
// 0: channel from JPEG-encoded image
// 1: gray scale
// 3: RGB image
const TENSORFLOW_CHANNEL = 3;

export const getTfModel = async () => {
  try {
    // wait until tensorflow is ready
    await tf.ready();
    // load the trained model
    return await tf.loadGraphModel(bundleResourceIO(modelJson, weights));
  } catch (error) {
    console.log('Could not load model', error);
  }
};

export const convertBase64ToTensor = async props => {
  try {
    const uIntArray = Base64Binary.decode(props.base64);
    // decode a JPEG-encoded image to a 3D Tensor of dtype
    const decodedImage = decodeJpeg(uIntArray, 3);
    // reshape Tensor into a 4D array
    return decodedImage.reshape([
      1,
      BITMAP_DIMENSION,
      BITMAP_DIMENSION,
      TENSORFLOW_CHANNEL,
    ]);
  } catch (error) {
    console.log('Could not convert base64 string to tesor', error);
  }
};

export const getPrediction = async (model, tensor) => {
  const reshapedTensor = tf.reshape(tensor, [
    1,
    BITMAP_DIMENSION,
    BITMAP_DIMENSION,
    TENSORFLOW_CHANNEL,
  ]);
  try {
    // predict against the model
    return await model.executeAsync(tensor);
    // return typed array
  } catch (error) {
    console.error('error', error);
  }
};

const stop = false;
const frameCount = 0;
let fps, fpsInterval, startTime, now, then, elapsed;

// initialize the timer variables and start the animation

function startAnimating(fps) {
  fpsInterval = 1000 / fps;
  then = Date.now();
  startTime = then;
  animate();
}

// the animation loop calculates time elapsed since the last loop
// and only draws if your specified fps interval is achieved

function animate() {
  // request another frame

  requestAnimationFrame(animate);

  // calc elapsed time since last loop

  now = Date.now();
  elapsed = now - then;

  // if enough time has elapsed, draw the next frame

  if (elapsed > fpsInterval) {
    // Get ready for next frame by setting then=now, but also adjust for your
    // specified fpsInterval not being a multiple of RAF's interval (16.7ms)
    then = now - (elapsed % fpsInterval);

    // Put your drawing code here
  }
}
