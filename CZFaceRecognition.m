(* ::Package:: *)

CZFaceRecognitionDatabase=CloudImport["FaceRecognitionDatabase"];


cl=Classify[CZFaceRecognitionDatabase,IndeterminateThreshold->.5,Method->"NearestNeighbors"];


CZClassifier[ desc_ ] := Module[{dists=EuclideanDistance[#,desc]&/@CZFaceRecognitionDatabase[[All,1]]},If[Min[dists]>60,"?",CZFaceRecognitionDatabase[[First@Ordering[dists,1],2]]]]


CZHighlightFaces[ image_Image ] :=
   HighlightImage[
      img,
      {#["Gender"] /. "Male"->Blue /. "Female"->Pink,#["BoundingBox"],Inset[Style[CZClassifier[#["Descriptor"]],White,
         FontSize->Scaled[1/50],Background->Black],First[#["BoundingBox"]],{Left,Bottom}]}&/@FindFaces[img,{"BoundingBox","Descriptor","Gender"}]];
