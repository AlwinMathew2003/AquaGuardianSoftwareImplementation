import express from 'express';
import mongoose from "mongoose"
import bcrypt from 'bcryptjs'
import { User } from './model/UserDetails.js'
import cors from 'cors'
import jwt from "jsonwebtoken";

const app = express();
app.use(express.json());
app.use(cors());


const JWT_SECRET =
  "hvdvay6ert72839289()aiyg8t87qt72393293883uhefiuh78ttq3ifi78272jdsds039[]]pou89ywe";

mongoose.connect('mongodb+srv://akhiljose:QLeXvX9q96D9m2mo@carapp.ntg3p.mongodb.net/?retryWrites=true&w=majority&appName=CarAPP')
        .then(() => console.log('connected to database'))
        .catch( ()=>console.log("not connected"))




app.get("/", (req, res) => {
  res.send({ status: "Started" });
});

app.post("/register", async (req, res) => {
  const { name, email, mobile, password } = req.body;
  console.log(req.body);

  const oldUser = await User.findOne({ email: email });

  if (oldUser) {
    return res.send({ data: "User already exists!!" });
  }
  const encryptedPassword = await bcrypt.hash(password, 10);

  try {
    await User.create({
      name: name,
      email: email,
      mobile,
      password: encryptedPassword,
    });
    res.send({ status: "ok", data: "User Created" });
  } catch (error) {
    res.send({ status: "error", data: error });
  }
});

app.post("/login-user", async (req, res) => {
  const { email, password } = req.body;
  console.log(req.body);
  const oldUser = await User.findOne({ email: email });

  if (!oldUser) {
    return res.status(404).json({ message: "User doesn't exist!!" });
  }

  if (await bcrypt.compare(password, oldUser.password)) {
    const token = jwt.sign({ email: oldUser.email }, JWT_SECRET);
    console.log(token);
    return res.status(200).send({
      status: "ok",
      data: token,
    });
  } else {
    return res.status(401).send({ error: "Invalid credentials" });
  }
});

app.post("/userdata", async (req, res) => {
  const { token } = req.body;
  try {
    const user = jwt.verify(token, JWT_SECRET);
    const useremail = user.email;

    User.findOne({ email: useremail }).then((data) => {
      return res.send({ status: "Ok", data: data });
    });
  } catch (error) {
    return res.send({ error: error });
  }
});

app.post("/updateuser", async (req, res) => {
  const {name, email, password, mobile} = req.body;
  console.log(req.body);
  try {
    await User.updateOne(
      { email: email },{$set: {name, email, password, mobile} }
    );
    return res.send({ status: "Ok", data: "Updated" });
  } catch (error) {
    return res.send({ error: error });
  }
});

app.get("/get-all-user", async (req, res) => {
  try {
    const data = await User.find({});
    res.send({ status: "Ok", data: data });
  } catch (error) {
    return res.send({ error: error });
  }
});

// app.post("/delete-user",async (req, res) => {
//  const {id}=req.body;
//  try {
//   await User.deleteOne({_id:id});
//   res.send({status:"Ok",data:"User Deleted"});
//  } catch (error) {
//   return res.send({ error: error });
  
//  }
// })



app.listen(5001, () => {
  console.log("Node js server started.");
});
