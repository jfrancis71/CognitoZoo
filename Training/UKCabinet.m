(* ::Package:: *)

(*
   UK Cabinet Minsters Jan 2018
   Note, not complete list.

 Source:
   http://metro.co.uk/2018/01/16/who-are-the-current-cabinet-ministers-and-who-also-attends-cabinet-meetings-7233875/
*)


faceImageBase = {
Import["https://static.independent.co.uk/s3fs-public/thumbnails/image/2017/06/04/10/theresa-may.png"]->"Theresa May",
Import["https://pbs.twimg.com/profile_images/665204710169792512/PgfbIAFq_400x400.jpg"]->"Philip Hammond",
Import["https://pbs.twimg.com/profile_images/490155751738847232/bs5sfeFR.jpeg"]->"Amber Rudd",
Import["https://www.telegraph.co.uk/content/dam/politics/2018/02/13/TELEMMGLPICT000154026824_trans_NvBQzQNjv4BqpVlberWd9EgFPZtcLiMQfy2dmClwgbjjulYfPTELibA.jpeg?imwidth=450"]->"Boris Johnson",
Import["https://pbs.twimg.com/profile_images/621956954819309568/ZcshnG_T.png"]->"David Davis",
Import["https://www.telegraph.co.uk/content/dam/news/2016/04/16/88000693_Mcc0067906ST_News.Interview_with_RT_Hon_Liam_Fox_MP-xlarge_trans_NvBQzQNjv4BqpiVx42joSuAkZ0bE9ijUnGH28ZiNHzwg9svuZLxrn1U.jpg"]->"Liam Fox",
Import["https://pbs.twimg.com/profile_images/748801720231161857/JkhsUaYO.jpg"]->"Michael Gove"
};


faceBase=Map[CZDLibFaceDescriptor[#[[1]]]->#[[2]]&,faceImageBase];


Export["c:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\CZModels\\FaceBaseUKCabinet.mx",faceBase];
