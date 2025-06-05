import os
import math
import warnings
import subprocess
import pandas as pd
import argparse  # Добавлен для обработки аргументов
from stone.image import DEFAULT_TONE_PALETTE

# Ignore all warnings
warnings.filterwarnings("ignore")

def convert_to_rgb(color_series):
    """Converts color values from hexadecimal to RGB."""
    cleaned_colors = color_series.str[1:]
    red = cleaned_colors.str[:2].apply(lambda x: int(x, 16))
    green = cleaned_colors.str[2:4].apply(lambda x: int(x, 16))
    blue = cleaned_colors.str[4:6].apply(lambda x: int(x, 16))
    return red, green, blue

def creation_von_lus_df():
    """Creates a DataFrame with RGB values based on the von Luschan scale."""
    data = {
        "type": list(range(1, 37)),
        "R": [244, 236, 250, 253, 253, 254, 250, 243, 244, 251,
              252, 254, 255, 255, 241, 239, 224, 242, 235, 235,
              227, 225, 223, 222, 199, 188, 156, 142, 121, 100,
              101, 96, 87, 64, 49, 27],
        "G": [242, 235, 249, 251, 246, 247, 240, 234, 241, 252,
              248, 246, 249, 249, 231, 226, 210, 226, 214, 217,
              196, 193, 193, 184, 164, 151, 107, 88, 77, 49,
              48, 49, 50, 32, 37, 28],
        "B": [245, 233, 247, 230, 230, 229, 239, 229, 234, 244,
              237, 225, 225, 225, 195, 173, 147, 151, 159, 133,
              103, 106, 123, 119, 100, 98, 67, 62, 48, 22,
              32, 33, 33, 21, 41, 46]
    }
    return pd.DataFrame(data)

def fitzpatrick_scale(skin_tone, default_palette):
    """Determines Fitzpatrick type based on skin tone color."""
    color_mapping = {
        (default_palette["color"][9], default_palette["color"][10]): 1,
        (default_palette["color"][7], default_palette["color"][8]): 2,
        (default_palette["color"][5], default_palette["color"][6]): 3,
        (default_palette["color"][3], default_palette["color"][4]): 4,
        (default_palette["color"][1], default_palette["color"][2]): 5,
        (default_palette["color"][0],): 6
    }
    for colors, index in color_mapping.items():
        if skin_tone in colors:
            return index
    raise ValueError("Error in determining Fitzpatrick type")

def eucl_metric(row, r_pic, g_pic, b_pic):
    """Calculates Euclidean distance between given and reference RGB values."""
    return math.sqrt((r_pic - row["R"]) ** 2 +
                     (g_pic - row["G"]) ** 2 +
                     (b_pic - row["B"]) ** 2)

def find_closest_rgb(df, r_1, g_1, b_1):
    """Finds the closest RGB color in the DataFrame to a given value."""
    distances = df.apply(lambda row: eucl_metric(row, r_1, g_1, b_1), axis=1)
    closest_index = distances.idxmin()
    min_value = distances.min()
    return min_value, closest_index

def equation_color(teg_color, percent_color):
    """Calculates the final RGB color based on tags and percentages."""
    sum_tone = [0, 0, 0]
    for teg, percent in zip(teg_color, percent_color):
        r, g, b = convert_to_rgb(pd.Series([teg]))
        sum_tone[0] += r.iloc[0] * percent
        sum_tone[1] += g.iloc[0] * percent
        sum_tone[2] += b.iloc[0] * percent
    return sum_tone

def von_lushan_scale(fitzpatrick_value, von_lus_df, sum_tone):
    """Finds the index on the von Luschan scale for a given Fitzpatrick type."""
    fitz_ranges = {
        1: (0, 6),
        2: (7, 13),
        3: (14, 20),
        4: (21, 27),
        5: (28, 34),
        6: (35, 36)
    }
    if fitzpatrick_value in fitz_ranges:
        start, end = fitz_ranges[fitzpatrick_value]
        return find_closest_rgb(von_lus_df[start:end + 1], *sum_tone)
    else:
        raise ValueError("Error in determining von Luschan type")

def linux_command(command):
    """Executes a Linux command in the terminal and outputs the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print("Command result:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Command execution error:", e.stderr)

def make_dir(path):
    """Creates a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def delete_file_if_exists(file_path):
    """Deletes the file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} deleted.")
    else:
        print(f"File {file_path} not found, deletion not required.")

def main():
    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description='Process skin tone classification.')
    parser.add_argument('--input', type=str, required=True, help='Path to directory with images')
    parser.add_argument('--output', type=str, required=True, help='Path for output results')
    args = parser.parse_args()

    images_path = args.input
    results_path = args.output
    result_file = f"{results_path}/result.csv"

    # Создаем директории для результатов
    make_dir(results_path)
    delete_file_if_exists(result_file)

    # Проверка наличия изображений
    if not os.path.exists(images_path) or not os.listdir(images_path):
        print(f"No images found in {images_path}. Exiting.")
        return

    # Запуск обработки изображений
    command = f"stone --image_dirs {images_path} -o {results_path} -t 'color'"
    linux_command(command)

    if not os.path.exists(result_file):
        print(f"Result file {result_file} not found. Exiting.")
        return

    von_lus_df = creation_von_lus_df()
    df = pd.read_csv(result_file)
    print(f'Result file path: {os.path.abspath(result_file)}')

    extended_df = pd.DataFrame(columns=['photo', 'skin_tone_hex', 'von_lus_index', 'fitzpatrick_index'])

    for _, row in df.iterrows():
        photo_path = row['file']
        color_1 = row["skin tone"].lower()
        r_1, g_1, b_1 = convert_to_rgb(pd.Series([color_1]))
        accuracy_factor = row["accuracy(0-100)"] * 0.01
        r_1, g_1, b_1 = r_1.iloc[0] * accuracy_factor, g_1.iloc[0] * accuracy_factor, b_1.iloc[0] * accuracy_factor

        _, closest_index = find_closest_rgb(von_lus_df, r_1, g_1, b_1)
        fitzpatrick_index = fitzpatrick_scale(color_1, DEFAULT_TONE_PALETTE)

        print(f"Photo: {photo_path}")
        print(f"Luschan index: {closest_index}, Fitzpatrick index: {fitzpatrick_index}")

        new_row = pd.DataFrame([{
            'photo': photo_path,
            'skin_tone_hex': color_1,
            'von_lus_index': closest_index,
            'fitzpatrick_index': fitzpatrick_index
        }])

        extended_df = pd.concat([extended_df, new_row], ignore_index=True)

    save_path = f"{results_path}/skin_tone_results.csv"
    extended_df.to_csv(save_path, index=False)
    print(f"\nResults saved at {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()