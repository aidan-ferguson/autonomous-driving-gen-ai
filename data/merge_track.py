"""
This file provides a utility to merge multiple track samples with ground truth cones 
to be merged manually so that we get a complete track in with the origin at the point where
the LiDAR was turned on.

The CSV that we read has rows where each row is the position and colour of the cone, there are
spaces between lines which denotes that the measurements came from a separate frame.
"""

import argparse
import matplotlib.pyplot as plt

CONE_COLOUR_MAPPING = {
    "large_orange": "darkorange",
    "blue": "blue",
    "yellow": "yellow",
    "orange": "orange"
}

def main() -> None:
    # Command line args parser
    parser = argparse.ArgumentParser(description='Track segment merging script')
    parser.add_argument('filename', type=str, help='Specially formatted track segment CSV file')
    args = parser.parse_args()

    # Read all the track segments
    track_segments = []
    with open(args.filename, "r") as file:
        current_segment = []
        # Start from index 1 as we want to skip the header
        for line in file.readlines()[1:]:
            if all([c.isspace() for c in line]):
                # Track segment end - store segment if there are any cones
                if len(current_segment):
                    track_segments.append(current_segment)
                    current_segment = []
            else:
                # Cone position line, parse it and store the results
                cone = [value.strip() for value in line.split(",")]
                cone[:3] = [float(position) for position in cone[:3]]
                current_segment.append(cone)

    # Display a top down view of the cones in matplotlib and allow
    figure = plt.figure()
    plt.ion()
    plt.show()
    
    # Create a keyboard callback hander
    key_queue = []
    figure.canvas.mpl_connect('key_press_event', lambda event: key_queue.append(event))
    
    current_track = []
    for track_segment in track_segments:
        current_offset = [0.0, 0.0]

        # While user has not yet placed the track segment - listen for input and re-draw plot
        should_quit = False
        while not should_quit:
                print(key_queue)
                # Plot current track
                for cone in current_track:
                    plt.scatter(cone[0], cone[1], c=CONE_COLOUR_MAPPING[cone[3]])

                # Plot new track segment
                for cone in track_segment:
                    plt.scatter(cone[0], cone[1], c=CONE_COLOUR_MAPPING[cone[3]])

                # Process event queue
                for event in key_queue:
                    if event.key == 'esc' or event.key == 'enter':
                        should_quit = True
                    key_queue = []

                # Draw to screen
                plt.draw()
                plt.pause(0.001)

if __name__ == "__main__":
    main()