import {InferenceSession} from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import * as tf from '@tensorflow/tfjs';

const importURL =
  'https://github.com/a-maghan/Pytorch-model-converion/raw/main/converted_model/onnx/model.onnx';

export const getOnnxModel = async () => {
  const path = `${RNFS.DocumentDirectoryPath}/model.onnx`;
  if (!(await RNFS.exists(path))) {
    await RNFS.downloadFile({
      fromUrl: importURL,
      toFile: path,
    }).promise.then(p => {
      console.log('written');
    });
  }
  console.log('BLAH EXISTS');
  try {
    return await InferenceSession.create(path);
  } catch (error) {
    console.error('error while loading model', error);
  }
};
export const runOnnxModel = async (session, tensor) => {
  const reshapedTensor = tf.reshape(tensor, [1, 3, 8, 224, 224]);
  console.log('reshapedTensor', reshapedTensor);
  try {
    // predict against the model
    return await session.run(reshapedTensor);
  } catch (error) {
    console.log('Error predicting from tesor image', error);
  }
};
