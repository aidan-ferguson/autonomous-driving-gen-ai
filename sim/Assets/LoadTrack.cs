using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;

public class LoadTrack : MonoBehaviour
{
    public GameObject yellowConePrefab;
    public GameObject blueConePrefab;
    public GameObject largeOrangePrefab;
    public GameObject orangePrefab;
    public GameObject carPrefab;
    public string csvFilePath; // The path to the CSV file
    // public float scaleFactor = 1f; // Scale factor for adjusting the size of instantiated objects

    void Start()
    {
        LoadCSV();
    }

    void LoadCSV()
    {
        Vector3 ground_plane_pos = GameObject.Find("ground_plane").transform.position;

        string[] csvLines = File.ReadAllLines(csvFilePath);

        // Skip the first line to avoid the header
        bool firstLine = true;
        foreach (string line in csvLines)
        {
            if(firstLine)
            {
                firstLine = false;
            }
            else
            {
                string[] values = line.Split(',');

                string type = values[0];
                float x = float.Parse(values[1]);
                float y = float.Parse(values[2]);

                Vector3 position = new Vector3(x, ground_plane_pos.y, y);

                // Instantiate object at the specified position
                GameObject obj = null;
                if (type == "blue")
                {
                    // Note, all small cones must be offset by a quarter of their height
                    position.y += 0.03f;
                    obj = Instantiate(blueConePrefab, position, Quaternion.Euler(-90, 0, 0));
                    obj.tag = "blue_cone";
                }
                else if(type == "yellow")
                {
                    position.y += 0.03f;
                    obj = Instantiate(yellowConePrefab, position, Quaternion.Euler(-90, 0, 0));
                    obj.tag = "yellow_cone";
                }
                else if(type == "orange")
                {
                    position.y += 0.03f;
                    obj = Instantiate(orangePrefab, position, Quaternion.Euler(-90, 0, 0));
                    obj.tag = "orange_cone";
                }
                else if(type == "big_orange")
                {
                    obj = Instantiate(largeOrangePrefab, position, Quaternion.Euler(-90, 0, 0));
                    obj.tag = "large_orange_cone";
                }
                else if(type == "car_start")
                {
                    // Create car at start, note car must be
                    Instantiate(carPrefab, position, Quaternion.identity);
                    // GameObject.Find("Main Camera").transform.position = position;
                }
                
                if (obj != null)
                {
                    // Add physics to the object, note isKinematic is required as we have non-convex hulls
                    Rigidbody rigidBody = obj.AddComponent<Rigidbody>();
                    rigidBody.isKinematic = true;

                    // Add a concave mesh collider, note must be non-convex to match the shape of the cone
                    MeshCollider meshCollider = obj.AddComponent<MeshCollider>();
                    meshCollider.convex = false;
                }
            }
        }
    }
}