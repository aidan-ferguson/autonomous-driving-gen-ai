using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

public class SemanticSegmentation : MonoBehaviour
{
    public int imageWidth = 640;
    public int imageHeight = 480;
    public bool captureImage = false;
    public int frameCounter = 0;
    public string outputFolder = "";
    public RenderTexture renderTexture = null;

    private Camera cam;
    private Texture2D segmentationImage;

    // Colour definitions should be in-line with those found in fsoco_parser.py
    // Note, the colours in the python script are in RGB [0,255] format, these are in RGBA [0,1]
    private Color yellow_colour = new Color(255.0f/255.0f, 255.0f/255.0f, 0.0f/255.0f, 1.0f);
    private Color blue_colour = new Color(0.0f/255.0f, 0.0f/255.0f, 255.0f/255.0f, 1.0f);
    private Color orange_colour = new Color(255.0f/255.0f, 165.0f/255.0f, 0.0f/255.0f, 1.0f);
    private Color large_orange_colour = new Color(255.0f/255.0f, 69.0f/255.0f, 0.0f/255.0f, 1.0f);
    


    void Start()
    {
        cam = GetComponent<Camera>();
        segmentationImage = new Texture2D(imageWidth, imageHeight);
    }

    void Update()
    {
        if(captureImage)
        {
            if (outputFolder.Length != 0)
            {
                // Save the rasterised camera view
                CamCapture();
                // Get and save a segmentation mask
                RenderSegmentationImage();
                // Get all the bounding boxes in the frame
                SaveBoundingBoxes();
                captureImage = false;
                frameCounter++;
            }
        }
    }

    double clamp(double val, double min, double max)
    {
        return Math.Min(Math.Max(val, min), max);
    }

    void CamCapture()
    {
        // The Render Texture in RenderTexture.active is the one
        // that will be read by ReadPixels.
        var currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;

        // Render the camera's view.
        cam.Render();

        // Make a new texture and read the active Render Texture into it.
        Texture2D image = new Texture2D(renderTexture.width, renderTexture.height);
        image.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        image.Apply();

        // Replace the original active Render Texture.
        RenderTexture.active = currentRT;
 
        System.IO.File.WriteAllBytes($"{outputFolder}/frame_{frameCounter}.png", image.EncodeToPNG());
    }

    void SaveBoundingBoxes()
    {
        List<GameObject> cones = new List<GameObject>();
        foreach(string tag in (new[] {"yellow_cone", "blue_cone", "large_orange_cone", "orange_cone"}))
        {
            foreach(GameObject obj in GameObject.FindGameObjectsWithTag(tag))
            {
                cones.Add(obj);
            }
        }

        List<string> yolo_annotation = new List<string>();

        foreach(GameObject cone in cones)
        {
            // Using the box collider calculate the 3D->2D projections for each corner and fit a BB around them
            double min_x = double.PositiveInfinity;
            double min_y = double.PositiveInfinity;
            double max_x = double.NegativeInfinity;
            double max_y = double.NegativeInfinity;

            // TODO: if the mesh stuff works, we can remove the multiple colliders
            Mesh mesh = cone.GetComponent<MeshFilter>().mesh;
            foreach(Vector3 local_vertex in mesh.vertices)
            {
                Vector3 vertex = cone.transform.TransformPoint(local_vertex);

                // GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                // sphere.transform.position = vertex;
                // sphere.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
                Vector3 screen_point = cam.WorldToViewportPoint(vertex);

                screen_point.x *= imageWidth;
                screen_point.y *= imageHeight;

                // Go from (0,0) at bottom left to top left
                screen_point.y = imageHeight - screen_point.y;
                if(screen_point.z < 0)
                {
                    continue;
                }
                if (screen_point.x < min_x)
                {
                    min_x = screen_point.x;
                }
                if (screen_point.x > max_x)
                {
                    max_x = screen_point.x;
                }
                if (screen_point.y > max_y)
                {
                    max_y = screen_point.y;
                }
                if (screen_point.y < min_y)
                {
                    min_y = screen_point.y;
                }
            }

            // If box fully outside of image, skip this iteration
            if(!(
                (max_x < 0) || (max_y < 0) || (min_y > imageHeight) || (min_x > imageWidth) ||
                ((min_x < 0) && (min_y < 0) && (max_x > imageWidth) && (max_y > imageHeight))
            ))
            {
                // We may still have some bounding box corners outside of the screen, clamp all values to a valid range
                min_x = clamp(min_x, 0, imageWidth);
                max_x = clamp(max_x, 0, imageWidth);
                min_y = clamp(min_y, 0, imageHeight);
                max_y = clamp(max_y, 0, imageHeight);

                // Now we have our bounding box in min_x, ...
                // Save to array of strings in YOLO format
                min_x /= imageWidth;
                max_x /= imageWidth;
                min_y /= imageHeight;
                max_y /= imageHeight;

                // Create YOLO annotation line
                /*
                    Note
                    blue = 0
                    orange = 1
                    large_orange = 2
                    yellow = 3
                */
                int class_id = 0;
                if(cone.tag == "blue_cone")
                {
                    class_id = 0;
                }
                else if(cone.tag == "yellow_cone")
                {
                    class_id = 3;
                }
                else if(cone.tag == "large_orange_cone")
                {
                    class_id = 2;
                }
                else if(cone.tag == "orange_cone")
                {
                    class_id = 1;
                }

                string yolo_line = $"{class_id} {(max_x+min_x)/2} {(max_y+min_y)/2} {max_x-min_x} {max_y-min_y}";
                yolo_annotation.Add(yolo_line);
            }
        }

        System.IO.File.WriteAllText($"{outputFolder}/frame_{frameCounter}_yolo.txt", string.Join("\n", yolo_annotation));
    }

    void RenderSegmentationImage()
    {
        for (int x = 0; x < imageWidth; x++)
        {
            for (int y = 0; y < imageHeight; y++)
            {
                Ray ray = cam.ScreenPointToRay(new Vector3(x, y, 0));
                RaycastHit hit;
                
                if (Physics.Raycast(ray, out hit))
                {
                    // Black pixel by default
                    Color pixelColour = Color.black;

                    string tag = hit.collider.tag;
                    if ((tag.Length > 0) && (tag != "Untagged"))
                    {
                        // Object has tag, check if it is a cone and colour it appropriately
                        if (tag == "yellow_cone")
                            pixelColour = yellow_colour;
                        else if(tag == "blue_cone")
                            pixelColour = blue_colour;
                        else if(tag == "large_orange_cone")
                            pixelColour = large_orange_colour;
                        else if(tag == "orange_cone")
                            pixelColour = orange_colour;
                    }

                    // Set the pixel colour
                    segmentationImage.SetPixel(x, y, pixelColour);
                }
                else
                {
                    // No hit, probably into skybox - return black
                    segmentationImage.SetPixel(x, y, Color.black);
                }
            }
        }

        byte[] bytes = segmentationImage.EncodeToPNG();
        System.IO.File.WriteAllBytes($"{outputFolder}/frame_{frameCounter}_mask.png", bytes);
    }

    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        // We want to render to texture AND the screen, this will copy from the texture to the screen
        Graphics.Blit(src, renderTexture);
        Graphics.Blit(src, (RenderTexture)null);
    }

}