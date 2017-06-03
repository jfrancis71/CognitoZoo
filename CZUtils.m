(* ::Package:: *)

CZIntersection[a_, b_] := Module[{xa=Max[a[[1,1]],b[[1,1]]],ya=Max[a[[1,2]],b[[1,2]]],xb=Min[a[[2,1]],b[[2,1]]],yb=Min[a[[2,2]],b[[2,2]]]},
   If[xa>xb||ya>yb,0,(xb-xa+1)*(yb-ya+1)]]
CZArea[a_] := ( a[[1,1]]-a[[2,1]] ) * ( a[[1,2]]-a[[2,2]] )
CZUnion[a_,b_] := CZArea[a] + CZArea[b] - CZIntersection[a, b]


(* Had considered using RegionIntersection/RegionUnion but this was overly general and unacceptably slow in practice.
   Not uncommon to see 100 raw detections, hence 10,000 pairs to evaluate.
*)
CZIntersectionOverUnion[a_, b_]:= 
   CZIntersection[ a, b ] / CZUnion[a, b]


CZDeleteOverlappingWindows[ {} ] := {};
CZDeleteOverlappingWindows[ objects_ ] :=
   Extract[objects,
      Position[
         Total[Table[
         Table[If[CZIntersectionOverUnion[objects[[a,2;;3]],objects[[b,2;;3]]]>.25&&objects[[a,1]]<objects[[b,1]],1,0],{b,1,Length[objects]}]
            ,{a,1,Length[objects]}]],
         0]][[All,2;;3]]
