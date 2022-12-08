import React, {useEffect, useState} from 'react';
import {SafeAreaView, StatusBar, Text} from 'react-native';
import {Colors} from 'react-native/Libraries/NewAppScreen';

import * as tf from '@tensorflow/tfjs';
import {cameraWithTensors} from '@tensorflow/tfjs-react-native';
import {Camera} from 'expo-camera';

import styles from './styles';
import {getTfModel, getPrediction} from './tfhelper';

// Tensor Camera replaces Expo Camera
const TensorCamera = cameraWithTensors(Camera);
// Size of the Tensors
// Fixed to 224,224,3 for input into custom model
const tensorDims = {height: 224, width: 224, depth: 3};

const ExpoCamera = () => {
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
        const loadedModel = await getTfModel();
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
      const prediction = await getPrediction(model, tensor);
      console.log('prediction', prediction);
      tf.dispose(tensor);
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
          resizeHeight={tensorDims.height}
          resizeWidth={tensorDims.width}
          resizeDepth={tensorDims.depth}
          autorender={true}
          onReady={imageAsTensors => handleCameraStream(imageAsTensors)}
          // onReady={() => {}}
        />
      )}
    </SafeAreaView>
  );
};

export default ExpoCamera;
