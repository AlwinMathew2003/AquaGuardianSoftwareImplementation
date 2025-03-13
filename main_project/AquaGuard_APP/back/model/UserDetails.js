import mongoose from "mongoose"

const UserDetailSchema = new mongoose.Schema(
  {
    name: String,
    email: { type: String, unique: true },
    mobile: String,
    password: String,
    // image:String,
    // gender:String,
    // profession:String,
    // userType:String
    
  }
);

export const User = mongoose.model("UserInfo", UserDetailSchema);
