import {Router} from "express"
const router = Router()
import { upload } from "../middlewares/multer.middleware.js";
import { 
    registerUser,
    loginUser ,
    logoutUser} 
from "../controllers/user.controller.js";
import { verifyJWT } from "../middlewares/auth.middleware.js";

router.route("/register").post(
    upload.single("avatar"),
    registerUser);
router.route("/login").post(loginUser);

//secured routes
router.route("/logout").post(verifyJWT,logoutUser);


export default router;