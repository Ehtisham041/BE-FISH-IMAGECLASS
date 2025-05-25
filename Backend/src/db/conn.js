import mongoose from "mongoose";
import { DB_NAME } from "../constants.js";



const connectDB = async () => {
  try {
    const uri = `${process.env.MONGODB_URI}/${DB_NAME}?retryWrites=true&w=majority`;

    const conn = await mongoose.connect(uri, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });

    console.log(`✅ MongoDB Connected: ${conn.connection.host}`);
  } catch (error) {
    console.error("❌ MongoDB connection error:", error);
    process.exit(1);
  }
};

export default connectDB;
