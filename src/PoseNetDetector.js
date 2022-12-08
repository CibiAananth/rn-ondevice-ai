import React, {useEffect, useState} from 'react';
import {SafeAreaView, StatusBar, Text} from 'react-native';
import {Colors} from 'react-native/Libraries/NewAppScreen';

import * as tf from '@tensorflow/tfjs';
import {cameraWithTensors} from '@tensorflow/tfjs-react-native';
import {Camera} from 'expo-camera';

import styles from './styles';
import {
  inputTensorHeight,
  inputTensorWidth,
  loadPosenetModel,
  runPosenetModel,
} from './posenetHelper';

// Tensor Camera replaces Expo Camera
const TensorCamera = cameraWithTensors(Camera);

const PosenetDetector = () => {
  const [frameworkReady, setFrameworkReady] = useState(false);
  const [hasPermission, setHasPermission] = useState(null);
  const [model, setModel] = useState(null);

  let requestAnimationFrameId = 0;

  useEffect(() => {
    if (!frameworkReady) {
      (async () => {
        //check permissions
        const {status} = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === 'granted');
        console.log('waiting modal load');
        await tf.ready();
        const loadedModel = await loadPosenetModel();
        setModel(loadedModel);
        setFrameworkReady(true);
        console.log('modal loaded');
      })();
    }
  }, [frameworkReady]);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(requestAnimationFrameId);
    };
  }, [requestAnimationFrameId]);

  let counter = 1;
  const handleCameraStream = imageAsTensors => {
    const loop = async () => {
      const tensor = imageAsTensors.next().value;
      const prediction = await runPosenetModel(model, tensor);
      console.log('prediction', prediction);
      tf.dispose([tensor]);
      if (counter > 10) {
        cancelAnimationFrame(requestAnimationFrameId);
        return;
      }
      counter++;
      requestAnimationFrameId = requestAnimationFrame(loop);
    };
    loop();
  };

  return (
    <SafeAreaView style={{backgroundColor: Colors.darker}}>
      <StatusBar barStyle={'dark-content'} backgroundColor={Colors.darker} />
      <Text style={styles.sectionTitle}>Welcome to TF camera demo</Text>
      {hasPermission && frameworkReady && (
        <TensorCamera
          style={styles.camera}
          type={Camera.Constants.Type.back}
          zoom={0}
          cameraTextureHeight={1920}
          cameraTextureWidth={1080}
          resizeHeight={inputTensorHeight}
          resizeWidth={inputTensorWidth}
          resizeDepth={3}
          autorender={true}
          onReady={imageAsTensors => handleCameraStream(imageAsTensors)}
          // onReady={() => {}}
        />
      )}
    </SafeAreaView>
  );
};

export default PosenetDetector;
