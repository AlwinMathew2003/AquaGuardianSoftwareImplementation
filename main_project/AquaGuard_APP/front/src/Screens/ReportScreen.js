import { StyleSheet, Text, View, ScrollView, Pressable } from "react-native";
import React from "react";
import { LinearGradient } from "expo-linear-gradient";
import { Feather, Entypo, Ionicons, Octicons } from "@expo/vector-icons";
import { MaterialCommunityIcons } from "@expo/vector-icons";

function ReportScreen() {
  return (
    <ScrollView>
      <LinearGradient colors={["#7F7FD5", "#E9E4F0"]} style={{ flex: 1 }}>
        <View style={styles.container}>
          {/* Top Bar */}
          <View style={styles.topBar}>
            <Feather name="bar-chart" size={24} color="black" />
            <Text style={styles.title}>Water Quality Analysis</Text>
            <Entypo name="lock" size={24} color="black" />
          </View>

          {/* Employee and Attendance Buttons */}
          <View style={styles.buttonsRow}>
            <Pressable
              // onPress={() => router.push("/(home)/employees")}
              style={styles.button}
            >
              <View style={styles.iconWrapper}>
                <Ionicons name="ios-people-sharp" size={24} color="black" />
              </View>
              <Text style={styles.buttonText}>River List</Text>
            </Pressable>

            <Pressable
              // onPress={() => router.push("/(home)/markattendance")}
              style={styles.button}
            >
              <View style={styles.iconWrapper}>
                <Ionicons name="ios-people-sharp" size={24} color="black" />
              </View>
              <Text style={styles.buttonText}>Accuracy</Text>
            </Pressable>
          </View>

          {/* Reports Section */}
          <View style={styles.reportsContainer}>
            <Pressable
              // onPress={() => router.push("/(home)/attendance-report")}
              style={styles.reportItem}
            >
              <View style={styles.reportIcon}>
                <Ionicons name="newspaper-outline" size={24} color="black" />
              </View>
              <Text style={styles.reportText}>Daily Reports</Text>
              <View style={styles.chevron}>
                <Entypo name="chevron-right" size={24} color="black" />
              </View>
            </Pressable>

            <Pressable
              // onPress={() => router.push("/(home)/summary")}
              style={styles.reportItem}
            >
              <View style={styles.reportIcon}>
                <Octicons name="repo-pull" size={24} color="black" />
              </View>
              <Text style={styles.reportText}>Weekly Reports</Text>
              <View style={styles.chevron}>
                <Entypo name="chevron-right" size={24} color="black" />
              </View>
            </Pressable>

            <Pressable
              // onPress={() => router.push("/(home)/all-reports")}
              style={styles.reportItem}
            >
              <View style={styles.reportIcon}>
                <Octicons name="report" size={24} color="black" />
              </View>
              <Text style={styles.reportText}>Monthly Reports</Text>
              <View style={styles.chevron}>
                <Entypo name="chevron-right" size={24} color="black" />
              </View>
            </Pressable>

            <Pressable
              // onPress={() => router.push("/(home)/overtime-employees")}
              style={styles.reportItem}
            >
              <View style={styles.reportIcon}>
                <Ionicons name="people" size={24} color="black" />
              </View>
              <Text style={styles.reportText}>All Generated Reports</Text>
              <View style={styles.chevron}>
                <Entypo name="chevron-right" size={24} color="black" />
              </View>
            </Pressable>
          </View>

          {/* Bottom Buttons */}
          <View style={styles.bottomButtonsRow}>
            <View style={styles.bottomButton}>
              <View style={styles.bottomButtonIcon}>
                <MaterialCommunityIcons
                  name="guy-fawkes-mask"
                  size={24}
                  color="black"
                />
              </View>
              <Text style={styles.bottomButtonText}>Pollution Criteria</Text>
            </View>

            <View style={styles.bottomButton}>
              <View style={styles.bottomButtonIcon}>
                <Feather name="bar-chart" size={24} color="black" />
              </View>
              <Text style={styles.bottomButtonText}>Increased pollution</Text>
            </View>
          </View>

          <View style={styles.bottomButtonsRow}>
            <View style={styles.bottomButton}>
              <View style={styles.bottomButtonIcon}>
                <MaterialCommunityIcons
                  name="guy-fawkes-mask"
                  size={24}
                  color="black"
                />
              </View>
              <Text style={styles.bottomButtonText}>Cost Savings</Text>
            </View>

            <View style={styles.bottomButton}>
              <View style={styles.bottomButtonIcon}>
                <Feather name="bar-chart" size={24} color="black" />
              </View>
              <Text style={styles.bottomButtonText}>Employee Performance</Text>
            </View>
          </View>
        </View>
      </LinearGradient>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 12,
  },
  topBar: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  title: {
    fontSize: 20,
    fontWeight: "600",
    color:'black'
  },
  buttonsRow: {
    marginTop: 20,
    flexDirection: "row",
    alignItems: "center",
    gap: 20,
  },
  button: {
    backgroundColor: "#D3CCE3",
    padding: 12,
    borderRadius: 6,
    alignItems: "center",
    justifyContent: "center",
    flex: 1,
  },
  iconWrapper: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: "white",
    alignItems: "center",
    justifyContent: "center",
  },
  buttonText: {
    marginTop: 7,
    fontWeight: "600",
  },
  reportsContainer: {
    marginTop: 20,
    backgroundColor: "white",
    paddingHorizontal: 10,
    paddingVertical: 10,
    borderRadius: 7,
  },
  reportItem: {
    backgroundColor: "#BE93C5",
    borderRadius: 6,
    padding: 10,
    flexDirection: "row",
    alignItems: "center",
    marginVertical: 10,
  },
  reportIcon: {
    padding: 7,
    width: 45,
    height: 45,
    borderRadius: 7,
    backgroundColor: "white",
    alignItems: "center",
    justifyContent: "center",
  },
  reportText: {
    marginLeft: 10,
    fontSize: 16,
    fontWeight: "600",
    flex: 1,
  },
  chevron: {
    width: 35,
    height: 35,
    borderRadius: 7,
    backgroundColor: "white",
    alignItems: "center",
    justifyContent: "center",
  },
  bottomButtonsRow: {
    marginTop: 20,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  bottomButton: {
    backgroundColor: "#f79d00",
    borderRadius: 6,
    padding: 12,
    alignItems: "center",
    justifyContent: "center",
    flex: 1,
  },
  bottomButtonIcon: {
    width: 35,
    height: 35,
    borderRadius: 7,
    backgroundColor: "white",
    alignItems: "center",
    justifyContent: "center",
  },
  bottomButtonText: {
    marginTop: 7,
  },
});

export default ReportScreen;
