(* ::Package:: *)

(*
   This functionality requires DLib. It includes the DLib face detector as well as the DLib Face Descriptor
   It has been tested on version 19.8.1
*)


DLibTmpFileName = FileNameJoin[{$TemporaryDirectory,"tmp.jpg"}];


DlibInitCmd="import dlib as db
import skimage.io
detector = db.get_frontal_face_detector()
sp = db.shape_predictor(r'"<>FileNameJoin[{$HomeDirectory,"shape_predictor_5_face_landmarks.dat"}]<>"')
facerec = db.face_recognition_model_v1(r'"<>FileNameJoin[{$HomeDirectory,"dlib_face_recognition_resnet_model_v1.dat"}]<>"')
";


sess = If[
   $OperatingSystem=="Windows",
   StartExternalSession[Association["System"->"Python","Executable"->"C:\\Users\\julian\\AppData\\Local\\Programs\\Python\\Python36\\python.exe"]],
   StartExternalSession["Python"]];


ExternalEvaluate[sess,DlibInitCmd]


CZDLibFaceDetection[file_String,height_Integer]:=
Map[{{#[[1,1]],height-#[[1,2]]},{#[[2,1]],height-#[[2,2]]}}&,
ExternalEvaluate[sess,"
img = skimage.io.imread(r'"<>DLibTmpFileName<>"')
dets = detector( img, 1 )
rects = []
for i,d in enumerate( dets ):
   rects.append( [ [ d.left(), d.bottom() ], [ d.right(), d.top() ] ] )
rects
"]]


CZDLibFaceDetection[image_Image]:=( 
   Export[DLibTmpFileName,image,IncludeMetaInformation->None];
   CZDLibFaceDetection["c:\\users\\julian\\tmp.jpg",ImageDimensions[image][[2]]]
)


angle[img_] := Module[{features = FacialFeatures[img,"NosePoints"]},
   If[Length[features]==1,
      ArcTan[(features[[1,{1,3}]][[1]]-features[[1,{1,3}]][[2]])[[1]],
         (features[[1,{1,3}]][[1]]-features[[1,{1,3}]][[2]])[[2]]],
      0.]
]


DLibFaceDescriptorCmd = "
img = skimage.io.imread(r'"<>DLibTmpFileName<>"')
dets = detector(img, 2)
descriptors = []
for k, d in enumerate( dets ):
   shape = sp(img, d)
   face_descriptor = facerec.compute_face_descriptor(img, shape)
   descriptors.append( face_descriptor )
#descriptors[0][0:1]
s = []
if ( len( descriptors ) > 0 ):
   for k, d in enumerate( descriptors[0] ):
      s.append( d )
s
";


CZDLibFaceDescriptor[ image_ ] := (
   Export[DLibTmpFileName, image];
   ExternalEvaluate[ sess, DLibFaceDescriptorCmd]
)


(* This is basically ImageTrim to extract subimage described by bounding box,
    but adds a 20% margin. Purpose: DLib face extract has quite tight bounding box around
    faces, makes it difficult to see hair etc. Note if it extends outside image then pads black
*)   
CZMarginImageTrim[image_,bb_]:=ImageTrim[image,bb,0.2*(bb[[2,1]]-bb[[1,1]]),Padding->Black]


CZMarginImageTrimList[image_,bb_]:=Map[CZMarginImageTrim[image,#]&,bb]


faceBase=Import["FaceBase.mx"];

CZDLibFaceRecognition[ image_Image ] := Module[
   {desc=CZDLibFaceDescriptor@image},
   If[ Length[ desc ] == 128,
      faceSpace=EuclideanDistance[desc,#]&/@faceBase[[All,1]];
      If[Min[faceSpace]>0.6,"Unknown",faceBase[[Ordering[faceSpace][[1]],2]]],
      "Unknown"]
   ]
