import React from 'react';
import {
  ImageBackground,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  View,
  Text,
  ScrollView,
  TouchableOpacity,
} from 'react-native';

import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import { useNavigation } from '@react-navigation/native';  // Import navigation hook
import COLORS from '../../assets/consts/color.js';

const Home = () => {
  const navigation = useNavigation(); // Access the navigation prop

  return (
    <SafeAreaView style={{flex: 1, backgroundColor: COLORS.white}}>
      <StatusBar translucent backgroundColor="rgba(0,0,0,0)" />
      
      <ScrollView contentContainerStyle={{flexGrow: 1}}>

        <ImageBackground
          style={{height: 300}}
          source={require('../../assets/AquaRobot.jpg')}>
        </ImageBackground>

        <View style={style.detailsContainer}>
          <View style={style.iconContainer}>
            <MaterialCommunityIcons name="water" color='#00BFFF' size={30} />
          </View>

          <View style={style.welcomeContainer}>
            <Text style={style.welcomeText}>Welcome</Text>
          </View>

          <Text style={style.aboutTitle}>About the Project</Text>
          <Text style={style.aboutDescription}>
            Our River Cleaning Robot is designed to remove waste from rivers and water bodies, ensuring cleaner environments. 
            Equipped with advanced sensors, it monitors water quality by measuring key parameters like pH levels and overall cleanliness. 
            This project aims to promote environmental sustainability and preserve the health of our waterways.
          </Text>
        </View>

        {/* Update TouchableOpacity for navigation */}
        <View style={style.footer}>
          <View style={{flex: 1, flexDirection: 'row', alignItems: 'center'}}>
            <Text style={style.footerText}>Go to</Text>
          </View>
          
          {/* TouchableOpacity for navigation */}
          <TouchableOpacity 
            style={style.bookNowBtn}
            onPress={() => navigation.navigate('LiveScreen')} // Navigate to the Control panel screen
          >
            <Text style={style.bookNowText}>
              Control panel
            </Text>
          </TouchableOpacity>
        </View>

      </ScrollView>
    </SafeAreaView>
  );
};

const style = StyleSheet.create({
  bookNowBtn: {
    height: 50,
    width: 150,
    backgroundColor: COLORS.white,
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  bookNowText: {
    color: COLORS.primary,
    fontSize: 16,
    fontWeight: 'bold',
  },
  iconContainer: {
    height: 60,
    width: 60,
    position: 'absolute',
    top: -30,
    backgroundColor: COLORS.white,
    borderRadius: 30,
    right: 20,
    elevation: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  detailsContainer: {
    paddingVertical: 40,
    paddingHorizontal: 20,
    backgroundColor: COLORS.white,
  },
  welcomeContainer: {
    flexDirection: 'row', 
    marginBottom: 20, 
    justifyContent: 'center',
  },
  welcomeText: {
    fontSize: 30,
    fontWeight: 'bold',
    color: COLORS.primary,
    textAlign: 'center',
  },
  aboutTitle: {
    marginTop: 20,
    fontWeight: 'bold',
    fontSize: 24,
    color: COLORS.primary,
  },
  aboutDescription: {
    marginTop: 15,
    lineHeight: 24,
    fontSize: 16,
    color: COLORS.dark,
    textAlign: 'justify', // Adds readability and structure
  },
  footer: {
    flexDirection: 'row',
    backgroundColor: COLORS.primary,
    height: 70,
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    borderTopLeftRadius: 15,
    borderTopRightRadius: 15,
  },
  footerText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.white,
  },
});

export {Home};
