import * as cocoSsd from '@tensorflow-models/coco-ssd';

export const inputTensorWidth = 152;
export const inputTensorHeight = 200;

export const loadCocoSsdModel = async () => {
  try {
    return await cocoSsd.load();
  } catch (error) {
    console.error('error while loading model', error);
  }
};

export const runCocoSsdModel = async (model, tensor) => {
  try {
    return await model.detect(tensor);
  } catch (error) {
    console.error('error while running model', error);
  }
};
// WARN  tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead
