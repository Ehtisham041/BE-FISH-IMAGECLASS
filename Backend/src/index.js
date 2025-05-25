import mongoose from "mongoose";
import connectDB from "./db/conn.js";
import dotenv from "dotenv";
dotenv.config();
import {app} from "./app.js"

connectDB()
.then(()=>{

 app.listen(process.env.PORT || 8000 ,()=>{
    console.log(`server is running on port ${process.env.PORT || 8000}`)
 })  
})
.catch((error)=>{
console.log("mongo db conn failed 'index.js' ",error);

})




















