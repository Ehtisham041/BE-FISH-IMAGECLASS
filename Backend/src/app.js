import express from "express";
import cors from "cors";
import cookieParser from "cookie-parser";
const app = express();

app.use(cors({
    origin: process.env.CORS_ORIGIN,
    credentials: true
}))
app.use(express.json());
app.use(express.urlencoded({extended:true , limit:"16kb"}));
app.use(express.static("public"));
app.use(cookieParser());
//routes 

//routes 
import userRouter from"./routes/user.routes.js";
app.use("/api/v1/users",userRouter);
//fish routes
import fishRouter from"./routes/fish.routes.js";
app.use("/api/v1/fish",fishRouter);
export {app};
