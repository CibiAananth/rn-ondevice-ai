import React, {useEffect, useState} from 'react';
import {ActivityIndicator, Text, View} from 'react-native';
import {Camera} from 'expo-camera';
import Svg, {Circle, Rect, G, Line} from 'react-native-svg';

import {cameraWithTensors} from '@tensorflow/tfjs-react-native';
import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';

import styles from './styles';
import {
  inputTensorHeight,
  inputTensorWidth,
  loadBlazeFaceModel,
  runBlazeFaceModel,
} from './blazenetHelper';

// Tensor Camera replaces Expo Camera
const TensorCamera = cameraWithTensors(Camera);

const PosenetDetector = () => {
  const [frameworkReady, setFrameworkReady] = useState(false);
  const [hasPermission, setHasPermission] = useState(null);
  const [model, setModel] = useState(null);
  const [fps, setFps] = useState(0);
  const [faces, setFaces] = useState([]);

  let requestAnimationFrameId = 0;

  useEffect(() => {
    if (!frameworkReady) {
      (async () => {
        //check permissions
        const {status} = await Camera.requestCameraPermissionsAsync();
        setHasPermission(status === 'granted');
        console.log('waiting modal load');
        await tf.ready();
        const loadedModel = await loadBlazeFaceModel();
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

  const handleCameraStream = imageAsTensors => {
    const loop = async () => {
      const tensor = imageAsTensors.next().value;
      const startTs = Date.now();
      setFaces(await runBlazeFaceModel(model, tensor));
      const latency = Date.now() - startTs;
      setFps(Math.floor(1000 / latency));
      tf.dispose([tensor]);
      requestAnimationFrameId = requestAnimationFrame(loop);
    };
    loop();
  };

  const renderPose = () => {
    const MIN_KEYPOINT_SCORE = 0.2;
    const {pose} = this.state;
    if (pose != null) {
      const keypoints = pose.keypoints
        .filter(k => k.score > MIN_KEYPOINT_SCORE)
        .map((k, i) => {
          return (
            <Circle
              key={`skeletonkp_${i}`}
              cx={k.position.x}
              cy={k.position.y}
              r="2"
              strokeWidth="0"
              fill="blue"
            />
          );
        });

      const adjacentKeypoints = posenet.getAdjacentKeyPoints(
        pose.keypoints,
        MIN_KEYPOINT_SCORE,
      );

      const skeleton = adjacentKeypoints.map(([from, to], i) => {
        return (
          <Line
            key={`skeletonls_${i}`}
            x1={from.position.x}
            y1={from.position.y}
            x2={to.position.x}
            y2={to.position.y}
            stroke="magenta"
            strokeWidth="1"
          />
        );
      });

      return (
        <Svg
          height="100%"
          width="100%"
          viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}>
          {skeleton}
          {keypoints}
        </Svg>
      );
    } else {
      return null;
    }
  };

  const renderFaces = () => {
    if (faces && faces.length) {
      const faceBoxes = faces.map((f, fIndex) => {
        const topLeft = f.topLeft;
        const bottomRight = f.bottomRight;

        const landmarks = f.landmarks.map((l, lIndex) => {
          return (
            <Circle
              key={`landmark_${fIndex}_${lIndex}`}
              cx={l[0]}
              cy={l[1]}
              r="2"
              strokeWidth="0"
              fill="blue"
            />
          );
        });

        return (
          <G key={`facebox_${fIndex}`}>
            <Rect
              x={topLeft[0]}
              y={topLeft[1]}
              fill={'red'}
              fillOpacity={0.2}
              width={bottomRight[0] - topLeft[0]}
              height={bottomRight[1] - topLeft[1]}
            />
            {landmarks}
          </G>
        );
      });

      const flipHorizontal = 1;
      return (
        <Svg
          height="100%"
          width="100%"
          viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}
          scaleX={flipHorizontal}>
          {faceBoxes}
        </Svg>
      );
    } else {
      return null;
    }
  };

  const camView = (
    <View style={styles.cameraContainer}>
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
      />
      <View style={styles.modelResults}>{renderFaces()}</View>
    </View>
  );

  console.log('fps', fps);

  return (
    <View style={{width: '100%'}}>
      {hasPermission && frameworkReady ? (
        camView
      ) : (
        <View style={[styles.loadingIndicator]}>
          <ActivityIndicator size="large" color="#FF0266" />
        </View>
      )}
    </View>
  );
};

export default PosenetDetector;
