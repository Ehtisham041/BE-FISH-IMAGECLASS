import { Router } from "express";
import { verifyJWT } from "../middlewares/auth.middleware.js";
import { upload } from "../middlewares/multer.middleware.js";
import { uploadFishAndPredict, getUserFishPredictions } from "../controllers/fish.controller.js";

const router = Router();
//fish routes
router.post("/upload", verifyJWT, upload.single("fishImage"), uploadFishAndPredict);
router.get("/my-predictions", verifyJWT, getUserFishPredictions);

export default router;
