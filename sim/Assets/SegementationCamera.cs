using UnityEngine;

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
                RenderSegmentationImage();
                captureImage = false;
            }
            else
            {
                Debug.LogWarning("Output folder not entered");
            }
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