import React, { useState } from 'react';
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  Image,
  ScrollView,
  TextInput,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import Slider from '@react-native-community/slider';
import COLORS from '../../assets/consts/color.js';
import { ShowToast } from '../components/Toast.js';

const LiveScreen = () => {
  const [remoteMode, setRemoteMode] = useState(true);
  const [ipAddress, setIpAddress] = useState('');
  const [isEditingIP, setIsEditingIP] = useState(true);
  const [motorSpeed, setMotorSpeed] = useState(0);
  const [ipAddress1, setIpAddress1] = useState('');
  const [isEditingIP1, setIsEditingIP1] = useState(true);

  const handleControlAction = async (action) => {
    if (ipAddress === '') {
      ShowToast('error', "Missing IP Address, Please provide an IP address");
    } else {
      try {
        const url = `http://${ipAddress}/${action}`;
        console.log(url);
        const response = await fetch(url);

        if (response.ok) {
          ShowToast('success', `Performed ${action}`);
        } else {
          ShowToast('error', `Failed to perform ${action}`);
        }
      } catch (error) {
        ShowToast('error', `Please provide valid IP address`);
      }
    }
  };

  const handleNavigation = async (action) => {
    if (ipAddress1 === '') {
      ShowToast('error', "Missing IP Address, Please provide an IP address");
    } else {
      try {
        const url = `http://${ipAddress1}/${action}`;
        console.log(url);
        const response = await fetch(url);

        if (response.ok) {
          ShowToast('success', `Performed ${action}`);
        } else {
          ShowToast('error', `Failed to perform ${action}`);
        }
      } catch (error) {
        ShowToast('error', `Please provide valid IP address for navigation`);
      }
    }
  };

  const updateMotorSpeed = async (speed) => {
    setMotorSpeed(speed);
    if (ipAddress === '') {
      ShowToast('error', "Missing IP Address, Please provide an IP address");
      return;
    }
    try {
      const url = `http://${ipAddress}/speed?value=${speed}`;
      console.log(url);
      const response = await fetch(url);

      if (response.ok) {
        ShowToast('success', `Speed set to ${speed}%`);
      } else {
        ShowToast('error', `Failed to set speed`);
      }
    } catch (error) {
      ShowToast('error', `Please provide valid IP address`);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar translucent backgroundColor="rgba(0,0,0,0)" />
      <ScrollView contentContainerStyle={styles.scrollViewContainer}>

        <View style={styles.imageContainer}>
          <Image source={require('../../assets/lake.jpg')} style={styles.topImage} />
        </View>

        <View style={styles.ipContainer}>
          <TextInput
            style={styles.ipInput}
            placeholder="Enter IP Address"
            value={ipAddress}
            editable={isEditingIP}
            onChangeText={setIpAddress}
          />
          <TouchableOpacity
            style={styles.editButton}
            onPress={() => setIsEditingIP(!isEditingIP)}
          >
            <Text style={styles.editButtonText}>
              {isEditingIP ? 'Save IP' : 'Edit IP'}
            </Text>
          </TouchableOpacity>
        </View>

        <View style={{ height: 20 }} /> 

        <View style={styles.ipContainer}>
          <TextInput
            style={styles.ipInput}
            placeholder="Enter IP Address for automation"
            value={ipAddress1}
            editable={isEditingIP1}
            onChangeText={setIpAddress1}
          />
          <TouchableOpacity
            style={styles.editButton}
            onPress={() => setIsEditingIP1(!isEditingIP1)}
          >
            <Text style={styles.editButtonText}>
              {isEditingIP1 ? 'Save IP' : 'Edit IP'}
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={[
              styles.modeButton,
              { backgroundColor: remoteMode ? COLORS.primary : COLORS.white },
            ]}
            onPress={() =>{setRemoteMode(true);handleNavigation("off")} } 
          >
            <Text style={{ color: remoteMode ? COLORS.white : COLORS.black, fontWeight: 'bold' }}>
              Remote
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.modeButton,
              { backgroundColor: !remoteMode ? COLORS.primary : COLORS.white },
            ]}
            onPress={() => {setRemoteMode(false);handleNavigation("on")} }
          >
            <Text style={{ color: !remoteMode ? COLORS.white : COLORS.black, fontWeight: 'bold' }}>
              Automatic
            </Text>
          </TouchableOpacity>
        </View>

        {remoteMode ? (
          <View style={styles.rowContainer}>
            <View style={styles.panel}>
              <View style={styles.controlsContainer}>
                <View style={styles.row}>
                  <TouchableOpacity
                    style={styles.upButton}
                    onPress={() => handleControlAction('forward')}
                  >
                    <Icon name="arrow-upward" size={24} color={COLORS.white} />
                  </TouchableOpacity>
                </View>

                <View style={styles.row}>
                  <TouchableOpacity
                    style={styles.leftButton}
                    onPress={() => handleControlAction('left')}
                  >
                    <Icon name="arrow-back" size={24} color={COLORS.white} />
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={styles.pauseButton}
                    onPress={() => handleControlAction('stop')}
                  >
                    <Icon name="pause" size={24} color={COLORS.white} />
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={styles.rightButton}
                    onPress={() => handleControlAction('right')}
                  >
                    <Icon name="arrow-forward" size={24} color={COLORS.white} />
                  </TouchableOpacity>
                </View>

                <View style={styles.row}>
                  <TouchableOpacity
                    style={styles.downButton}
                    onPress={() => handleControlAction('reverse')}
                  >
                    <Icon name="arrow-downward" size={24} color={COLORS.white} />
                  </TouchableOpacity>
                </View>

                <View style={styles.sliderContainer}>
                  <Text style={styles.sliderLabel}>Motor Speed: {motorSpeed}%</Text>
                  <Slider
                    style={styles.slider}
                    minimumValue={0}
                    maximumValue={100}
                    step={25}
                    value={motorSpeed}
                    onValueChange={(value) => setMotorSpeed(value)}
                    onSlidingComplete={(value) => updateMotorSpeed(value)}
                    minimumTrackTintColor={COLORS.primary}
                    maximumTrackTintColor={COLORS.grey}
                    thumbTintColor={COLORS.primary}
                  />
                </View>
              </View>
            </View>
          </View>
        ) : (
          <Text style={styles.automaticText}>Automatic mode enabled</Text>
        )}
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.black,
  },
  scrollViewContainer: {
    flexGrow: 1,
    paddingBottom: 20,
  },
  imageContainer: {
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.black,
  },
  topImage: {
    width: '100%',
    height: 350,
    resizeMode: 'contain',
    marginTop: -50,
  },
  ipContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginVertical: 15,
    marginTop: -35,
    paddingLeft: 12,
    paddingRight: 12,
  },
  ipInput: {
    backgroundColor: COLORS.white,
    color: COLORS.black,
    padding: 10,
    borderRadius: 8,
    borderColor: COLORS.primary,
    borderWidth: 1,
    flex: 1,
    marginRight: 20,
    marginBottom: 10,
  },
  editButton: {
    paddingHorizontal: 15,
    paddingVertical: 10,
    backgroundColor: COLORS.primary,
    borderRadius: 8,
  },
  editButtonText: {
    color: COLORS.white,
    fontWeight: 'bold',
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 20,
  },
  modeButton: {
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 25,
    marginHorizontal: 10,
    borderWidth: 1,
    borderColor: COLORS.primary,
  },
  rowContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    width: '100%',
  },
  panel: {
    flex: 1,
    alignItems: 'center',
    marginHorizontal: 10,
    borderColor: 'green',
    borderWidth: 2,
    borderRadius: 10,
    padding: 30,
    paddingBottom: 10,
  },
  controlsContainer: {
    alignItems: 'center',
  },
  row: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  upButton: {
    backgroundColor: COLORS.primary,
    width: 80,
    height: 80,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    margin: -21,
    marginLeft: -25,
  },
  leftButton: {
    backgroundColor: COLORS.primary,
    width: 80,
    height: 80,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 4,
    marginTop: 8,
  },
  pauseButton: {
    backgroundColor: COLORS.primary,
    width: 80,
    height: 80,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    margin: 7,
    marginRight: 10,
  },
  rightButton: {
    backgroundColor: COLORS.primary,
    width: 80,
    height: 80,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 4,
    marginTop: 8,
  },
  downButton: {
    backgroundColor: COLORS.primary,
    width: 80,
    height: 80,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    margin: -20,
  },
  automaticText: {
    fontSize: 18,
    color: 'purple',
    marginTop: 20,
    textAlign: 'center',
  },
  sliderContainer: {
    width: '90%',
    alignItems: 'center',
    marginTop: 20,
  },
  sliderLabel: {
    fontSize: 16,
    color: COLORS.dark,
    marginBottom: 10,
  },
  slider: {
    width: '100%',
    height: 40,
  },
});

export default LiveScreen;
