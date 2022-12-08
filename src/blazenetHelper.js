import * as blazeface from '@tensorflow-models/blazeface';

export const inputTensorWidth = 152;
export const inputTensorHeight = 200;

export const loadBlazeFaceModel = async () => {
  try {
    return await blazeface.load();
  } catch (error) {
    console.error('error while loading model', error);
  }
};

export const runBlazeFaceModel = async (model, tensor) => {
  try {
    return await model.estimateFaces(tensor, false);
  } catch (error) {
    console.error('error while running model', error);
  }
};
