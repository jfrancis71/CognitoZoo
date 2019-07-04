(* ::Package:: *)

<<CZDetectObjects.m


<<CZYolov3OpenImages.m


CZPrivacyRects[ image_ ]:= Union[
   CZDetectYoloOpenImages[image,DetectionClasses->{"Vehicle registration plate"},MaxOverlapFraction->0,AcceptanceThreshold->.005][[All,1]],
   FindFaces[image]]


replace[image_,rect_] := Raster[ImageData[GaussianFilter[ImageTrim[image,rect],Min[rect[[2,1]]-rect[[1,1]],rect[[2,2]]-rect[[1,2]]]/4],DataReversed->True],List@@rect]


CZHighlightPreservePrivacy[ image_Image ] := HighlightImage[ image, {replace[image,#]&/@CZPrivacyRects[ image ], CZDisplayObjects[ CZDetectObjects[ image, Method->"RetinaNet" ], CZCmap ] } ]
