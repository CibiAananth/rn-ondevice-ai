import * as posenet from '@tensorflow-models/posenet';

export const inputTensorWidth = 152;
export const inputTensorHeight = 200;
export const flipHorizontal = false;

export const loadPosenetModel = async () => {
  try {
    return await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: {width: inputTensorWidth, height: inputTensorHeight},
      multiplier: 0.75,
      quantBytes: 2,
    });
  } catch (error) {
    console.error('error while loading model', error);
  }
};

export const runPosenetModel = async (model, tensor) => {
  console.info('tensor', tensor);
  try {
    return await model.estimateSinglePose(tensor, {
      flipHorizontal,
    });
  } catch (error) {
    console.error('error while running model', error);
  }
};
