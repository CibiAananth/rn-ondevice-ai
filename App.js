import React, {useEffect, useState} from 'react';
import {SafeAreaView, StatusBar, StyleSheet, Text} from 'react-native';
import {Colors} from 'react-native/Libraries/NewAppScreen';

import * as tf from '@tensorflow/tfjs';
import {cameraWithTensors} from '@tensorflow/tfjs-react-native';
import {Camera} from 'expo-camera';

import sty from './src/styles';
import {getOnnxModel, runOnnxModel} from './onnx';

// Tensor Camera replaces Expo Camera
const TensorCamera = cameraWithTensors(Camera);
// Size of the Tensors
// Fixed to 224,224,3 for input into custom model
const tensorDims = {height: 224, width: 224, depth: 3};

const App = () => {
  const backgroundStyle = {
    backgroundColor: Colors.darker,
  };

  const [frameworkReady, setFrameworkReady] = useState(false);
  const [hasPermission, setHasPermission] = useState(null);
  const [model, setModel] = useState(null);

  let requestAnimationFrameId = 0;
  // useEffect(() => {
  //   if (!frameworkReady) {
  //     (async () => {
  //       //check permissions
  //       const {status} = await Camera.requestCameraPermissionsAsync();
  //       setHasPermission(status === 'granted');
  //       console.log('waiting modal load');
  //       const loadedModel = await getTfModel();
  //       setModel(loadedModel);
  //       console.log('modal loaded');
  //       setFrameworkReady(true);
  //     })();
  //   }
  // }, [frameworkReady]);

  useEffect(() => {
    (async () => {
      if (!frameworkReady) {
        //check permissions
        const {status} = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === 'granted');

        await tf.ready();

        const InferenceSession = await getOnnxModel();
        console.log('InferenceSession', InferenceSession);
        setModel(InferenceSession);
        setFrameworkReady(true);
      }
    })();
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
      const prediction = await runOnnxModel(model, tensor);
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
    <SafeAreaView style={backgroundStyle}>
      <StatusBar
        barStyle={'dark-content'}
        backgroundColor={backgroundStyle.backgroundColor}
      />
      <Text style={styles.sectionTitle}>{'isTfReady'}</Text>
      {hasPermission && frameworkReady && (
        <TensorCamera
          style={sty.camera}
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

const styles = StyleSheet.create({
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
    color: '#fff',
  },
  sectionDescription: {
    marginTop: 8,
    fontSize: 18,
    fontWeight: '400',
  },
  highlight: {
    fontWeight: '700',
  },
});

export default App;
