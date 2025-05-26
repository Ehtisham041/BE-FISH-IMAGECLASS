import { Router } from "express";
import { verifyJWT } from "../middlewares/auth.middleware.js";
import { upload } from "../middlewares/multer.middleware.js";
import { uploadFishAndPredict2 , uploadFishAndPredict, getUserFishPredictions } from "../controllers/fish.controller.js";

const router = Router();
//fish routes
router.post("/upload", verifyJWT, upload.single("fishImage"), uploadFishAndPredict);
router.post("/upload-predict2", upload.single("fishImage"), verifyJWT, uploadFishAndPredict2);

router.get("/my-predictions", verifyJWT, getUserFishPredictions);

export default router;
