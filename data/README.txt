untransformed_mesh.ply AND 3d_mesh_cropped.ply
    - original face mesh (of that guy from the spacetime faces)
    - one of these is the same face but many of the extra triangles have been cropped out so if i change the landmark points, its easy to redo later steps
    - POINTS picked w meshlab for that face: untransformed_mesh_landmark_points.pp
    - matrix transformation to get transformed_face_mesh.ply found using ipython notebook program... warp_3_to_2.ipynb

transformed_face_mesh.ply
    - this is a mesh that has been transformed so that it lines up (x and y coords) with the landmark points that i'm using for cropped faces
    - those landmarks are the points of: averageman_symmetric.txt ... but transformed a bit by -137.5 and -145 or something (warped->cropped)

template-aligned-colors-normals.ply AKA template-aligned-colors-normals.txt
    - resampled point cloud that lines up with the pixels in the cropped faces
    - ran this to get it: ./gridifyMesh data/transformed_face_mesh.txt data/cropped_mask2.png > thing.ply
    - had to make ply into a simple text file first...

Making a 3D shape
./shape3D --input  ~/Desktop/cflow/results_98/warpedimages.txt --templateMesh data/template-aligned-colors-normals.txt --visualize --ply ~/Desktop/neutral.ply