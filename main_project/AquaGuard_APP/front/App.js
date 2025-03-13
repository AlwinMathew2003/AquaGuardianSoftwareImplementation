import React, { useEffect, useState } from 'react';
import { Text, StyleSheet, View } from 'react-native';
import { Home } from './src/Screens/Home';
import LiveScreen from './src/Screens/LiveScreen.js';
import ReportScreen from './src/Screens/ReportScreen.js';
import Toast from 'react-native-toast-message'; // Import Toast
import { NavigationContainer, useNavigation, DrawerActions } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import 'react-native-gesture-handler';
import { createDrawerNavigator } from '@react-navigation/drawer';
import Icon from 'react-native-vector-icons/Entypo';
import DrawerContent from './DrawerContent.js';
import AsyncStorage from '@react-native-async-storage/async-storage';

const StackNav = () => {
  const Stack = createNativeStackNavigator();
  const navigation = useNavigation();

  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: '#04555c',
        },
        headerTintColor: 'white',
        headerTitleAlign: 'center',
      }}>
      <Stack.Screen
        name="Home"
        component={Home}
        options={{
          headerLeft: () => (
            <Icon
              name="menu"
              onPress={() => navigation.dispatch(DrawerActions.openDrawer())}
              size={30}
              color="white"
            />
          ),
        }}
      />
      <Stack.Screen name="LiveScreen" options={{ title: 'Live !!' }} component={LiveScreen} />
      <Stack.Screen name="ReportScreen" options={{ title: 'Reports' }} component={ReportScreen} />
    </Stack.Navigator>
  );
};

const DrawerNav = () => {
  const Drawer = createDrawerNavigator();

  return (
    <Drawer.Navigator
      drawerContent={(props) => <DrawerContent {...props} />}
      screenOptions={{
        headerShown: false,
      }}>
      <Drawer.Screen name="Home" component={StackNav} />
    </Drawer.Navigator>
  );
};

const App = () => {

  return (
    <NavigationContainer>
      <DrawerNav />
      <Toast />
    </NavigationContainer>
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#f0f0f0',
  },
  toggleText: {
    color: '#3498db',
    textAlign: 'center',
  },
  errorText: {
    color: 'red',
    marginTop: 16,
    textAlign: 'center',
  },
});

export default App;
