# BrickMapper

BrickMapper is a web application that helps LEGO enthusiasts identify and locate specific LEGO pieces within instruction manuals. The application allows users to search for LEGO elements by their ID and find which steps they appear in across different instruction booklets.

## Features

- **Element Search**: Search for LEGO elements by their unique element ID
- **Step Locator**: Find exactly which steps in which instruction booklets contain a specific element
- **PDF Links**: Direct links to the specific pages in official LEGO instruction PDFs
- **Element Details**: View information about elements including part number, name, color, and image
- **Data Management**: Upload CSV data to populate the database tables

## Technology Stack

- **Frontend**: React with Tailwind CSS for styling
- **Backend**: Supabase for database and authentication
- **Data Processing**: Papa Parse for CSV parsing and handling

## Database Schema

The application uses the following database tables:

- **elements**: Stores information about LEGO elements

  - `element_id`: Unique identifier for the element
  - `part_num`: The part number
  - `part_name`: The name of the part
  - `color_name`: The color name
  - `color_rgb`: The RGB value of the color
  - `img_url`: URL to an image of the element

- **sets**: Stores information about LEGO sets

  - `set_num`: Unique identifier for the set
  - `name`: The name of the set
  - `year`: The release year
  - `img_url`: URL to an image of the set

- **steps**: Stores information about where elements appear in instruction steps

  - `element_id`: Reference to the element
  - `set_num`: Reference to the set
  - `booklet_number`: The instruction booklet number
  - `page_number`: The page number in the booklet
  - `step_number`: The step number on the page

- **manuals**: Stores information about instruction manuals
  - `manual_id`: Unique identifier for the manual
  - `set_num`: Reference to the set
  - `booklet_number`: The booklet number

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   npm install
   ```
3. Create a `.env` file with your Supabase credentials:
   ```
   VITE_SUPABASE_URL=your_supabase_url
   VITE_SUPABASE_KEY=your_supabase_key
   ```
4. Start the development server:
   ```
   npm run dev
   ```

## Database Setup

1. Create the required tables in your Supabase project using the SQL Editor:

```sql
CREATE TABLE elements (
  element_id TEXT PRIMARY KEY,
  part_num TEXT,
  part_name TEXT,
  color_name TEXT,
  color_rgb TEXT,
  img_url TEXT
);

CREATE TABLE sets (
  set_num TEXT PRIMARY KEY,
  year INTEGER,
  img_url TEXT,
  name TEXT
);

CREATE TABLE steps (
  element_id TEXT,
  set_num TEXT,
  booklet_number INTEGER,
  page_number INTEGER,
  step_number INTEGER,
  PRIMARY KEY (element_id, set_num, booklet_number, page_number, step_number)
);

CREATE TABLE manuals (
  manual_id INTEGER PRIMARY KEY,
  set_num TEXT,
  booklet_number INTEGER
);
```

2. Configure Row Level Security (RLS) policies as needed for your application

## Data Import

The application supports importing data via CSV files. Prepare your CSV files with headers matching the column names in the database tables, then use the upload functionality in the application.

## Usage

1. Enter an element ID in the search box
2. View the element details and a list of steps where it appears
3. Click on page numbers to open the official LEGO instruction PDF at that specific page

## Troubleshooting

The application includes diagnostic tools to help troubleshoot database connection issues:

- **Run Diagnostics**: Checks database connection and table existence
- **Check Supabase Config**: Verifies that Supabase credentials are properly configured
