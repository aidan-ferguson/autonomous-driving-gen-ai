using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class SemanticSegmentation : MonoBehaviour
{
    public int imageWidth = 640;
    public int imageHeight = 480;
    public bool captureImage = false;
    public string outputFolder = "";

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
                // Get and save a segmentation mask
                RenderSegmentationImage();
                // Get all the bounding boxes in the frame
                SaveBoundingBoxes();
                captureImage = false;
            }
            else
            {
                Debug.LogWarning("Output folder not entered");
            }
        }
    }

    Vector3[] GetColliderVertexPositions(GameObject obj) {
            // This does not respect any rotations in the object and assumes it is lying on a flat plane
            Vector3[] vertices = new Vector3[8];
            Transform obj_transform = obj.transform;
        
            BoxCollider collider = obj.GetComponent<BoxCollider>();
            Vector3 sz = collider.size/2.0f;

            vertices[0] = obj_transform.TransformPoint(collider.center + new Vector3(sz.x, sz.y, sz.z));
            vertices[1] = obj_transform.TransformPoint(collider.center + new Vector3(-sz.x, sz.y, sz.z));
            vertices[2] = obj_transform.TransformPoint(collider.center + new Vector3(sz.x, -sz.y, sz.z));
            vertices[3] = obj_transform.TransformPoint(collider.center + new Vector3(-sz.x, -sz.y, sz.z));
            vertices[4] = obj_transform.TransformPoint(collider.center + new Vector3(sz.x, sz.y, -sz.z));
            vertices[5] = obj_transform.TransformPoint(collider.center + new Vector3(-sz.x, sz.y, -sz.z));
            vertices[6] = obj_transform.TransformPoint(collider.center + new Vector3(sz.x, -sz.y, -sz.z));
            vertices[7] = obj_transform.TransformPoint(collider.center + new Vector3(-sz.x, -sz.y, -sz.z));

            return vertices;
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

        foreach(GameObject cone in cones)
        {
            // Turn on box colliders and turn off mesh colliders temporarily
            cone.GetComponent<MeshCollider>().enabled = false;
            cone.GetComponent<BoxCollider>().enabled = true;

            // Using the box collider calculate the 3D->2D projections for each corner and fit a BB around them
            double min_x = double.PositiveInfinity;
            double min_y = double.PositiveInfinity;
            double max_x = double.NegativeInfinity;
            double max_y = double.NegativeInfinity;
            foreach(Vector3 vertex in GetColliderVertexPositions(cone))
            {
                GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.transform.position = vertex;
                sphere.transform.localScale = new Vector3(0.1f, 0.1f, 0.1f);
                Vector2 screen_point = cam.WorldToScreenPoint(vertex);
                
                // Go from (0,0) at bottom left to top left
                // screen_point.y = imageHeight - screen_point.y;
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

            // Now we have our bounding box in min_x, ...
            Debug.Log("found one");
            Debug.Log(cone.transform.position);
            Debug.Log(min_x);
            Debug.Log(min_y);
            Debug.Log(max_x - min_x);
            Debug.Log(max_y - min_y);

            cone.GetComponent<MeshCollider>().enabled = true;
            cone.GetComponent<BoxCollider>().enabled = false;
        }
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
        
        // segmentationImage.Apply();

        byte[] bytes = segmentationImage.EncodeToPNG();
        // if(!Directory.Exists(dirPath)) {
        //     Directory.CreateDirectory(dirPath);
        // }
        System.IO.File.WriteAllBytes(outputFolder + "/Image.png", bytes);
    }
}