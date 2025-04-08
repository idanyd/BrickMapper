import React, { useEffect, useState } from 'react';
import { createClient } from '@supabase/supabase-js';
import Papa from 'papaparse';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseKey = import.meta.env.VITE_SUPABASE_KEY;

const supabase = createClient(supabaseUrl, supabaseKey);
const BUILDING_INSTRUCTIONS_URL = "https://www.lego.com/en-gb/service/building-instructions";
const BUILDING_INSTRUCTIONS_PDFS_URL = "https://www.lego.com/cdn/product-assets/product.bi.core.pdf";
const App = () => {
  const [csvFile, setCsvFile] = useState(null);
  const [elementId, setElementId] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [elementDetails, setElementDetails] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    // Check if Supabase client is properly initialized
    console.log("Supabase URL:", supabaseUrl ? "Defined" : "Undefined");
    console.log("Supabase Key:", supabaseKey ? "Defined (value hidden)" : "Undefined");
    
    if (!supabaseUrl || !supabaseKey) {
      alert("Supabase configuration is missing. Please check your environment variables.");
      return;
    }
    
    // Log Supabase client to check if it's properly initialized
    console.log("Supabase client initialized:", supabase ? "Yes" : "No");
    
    runDiagnostics();
  }, []);

  const checkSupabaseConfig = () => {
    const configStatus = {
      url: supabaseUrl ? "✓ Defined" : "✗ Missing",
      key: supabaseKey ? "✓ Defined" : "✗ Missing",
      client: supabase ? "✓ Initialized" : "✗ Failed"
    };
    
    console.log("Supabase Configuration Status:", configStatus);
    
    alert(`
  Supabase Configuration Status:
  - URL: ${configStatus.url}
  - API Key: ${configStatus.key}
  - Client: ${configStatus.client}
  
  Check the console for more details.
    `);
  };

  const runDiagnostics = async () => {
    try {
      console.log("Running simplified diagnostics...");
      
      // 1. Check database connection with a simple query
      const { data: connectionTest, error: connectionError } = await supabase
        .from('elements')
        .select('*')
        .limit(1);
        
      if (connectionError) {
        console.error("Database connection issue:", connectionError);
        alert(`Database connection issue: ${connectionError.message || 'Unknown error'}`);
      } else {
        console.log("Database connection successful, elements table response:", connectionTest);
        
        if (connectionTest.length === 0) {
          console.log("Elements table exists but appears to be empty");
        }
      }
      
      // 2. Check each table individually with simple queries
      const tables = ['steps', 'sets', 'manuals'];
      
      for (const table of tables) {
        const { data, error } = await supabase
          .from(table)
          .select('*')
          .limit(1);
          
        if (error) {
          console.error(`Error accessing table '${table}':`, error);
          alert(`Table '${table}' may not exist or is not accessible: ${error.message || 'Unknown error'}`);
        } else {
          console.log(`Table '${table}' exists:`, data);
          
          if (data.length === 0) {
            console.log(`Table '${table}' appears to be empty`);
          }
        }
      }
      
      alert('Diagnostics complete. Check the console for detailed information.');
    } catch (error) {
      console.error('Diagnostics error:', error);
      alert(`Diagnostics error: ${error.message || 'Unknown error'}`);
    }
  };

  const fetchData = async () => {
    try {
      const [stepsResponse, elementsResponse] = await Promise.all([
        supabase.from('steps').select('*'),
        supabase.from('elements').select('*'),
      ]);

      if (stepsResponse.error) throw stepsResponse.error;
      if (elementsResponse.error) throw elementsResponse.error;

    } catch (error) {
      console.error('Error fetching data:', error.message);
    }
  };

  const handleCsvUpload = (e) => setCsvFile(e.target.files[0]);

  const uploadCsvData = async (tableName) => {
    if (!csvFile) {
      alert('Please upload a CSV file first.');
      return;
    }

    Papa.parse(csvFile, {
      header: true,
      complete: async ({ data }) => {
        try {
          const { error } = await supabase.from(tableName).insert(data);
          if (error) throw error;
          alert(`Data uploaded to ${tableName} successfully!`);
          fetchData();
        } catch (error) {
          console.error(`Error uploading data to ${tableName}:`, error.message);
        }
      },
      error: (error) => console.error('Error parsing CSV:', error.message),
    });
  };

  const handleSearch = async () => {
    if (!elementId) {
      alert('Please enter an Element ID to search.');
      return;
    }
  
    try {
      console.log("Searching for element ID:", elementId);
      
      // First try an exact match
      const { data: exactMatch, error: exactError } = await supabase
        .from('elements')
        .select('*')
        .eq('element_id', elementId.trim());
        
      if (exactError) {
        console.error("Error in exact search:", exactError);
        alert(`Error searching for element: ${exactError.message}`);
        return;
      }
      
      console.log("Exact match results:", exactMatch);
      
      if (exactMatch && exactMatch.length > 0) {
        // Found an exact match
        setElementDetails(exactMatch[0]);
        
        // Now search for steps
        const { data: stepsData, error: stepsError } = await supabase
          .from('steps')
          .select('*')
          .eq('element_id', exactMatch[0].element_id);
          
        if (stepsError) {
          console.error("Error searching steps:", stepsError);
          alert(`Error searching for steps: ${stepsError.message}`);
          return;
        }
        
        console.log("Steps data:", stepsData);
        
        if (!stepsData || stepsData.length === 0) {
          alert(`Element found but no steps contain this element.`);
          return;
        }
        
        // Get unique set numbers and booklet numbers from steps
        const setBookletPairs = [...new Set(stepsData.map(step => 
          `${step.set_num}|${step.booklet_number}`
        ))].map(pair => {
          const [set_num, booklet_number] = pair.split('|');
          return { set_num, booklet_number: parseInt(booklet_number) };
        });
        
        console.log("Set-Booklet pairs:", setBookletPairs);
        
        // Fetch manual IDs for each set-booklet pair
        const manualPromises = setBookletPairs.map(({ set_num, booklet_number }) => 
          supabase
            .from('manuals')
            .select('manual_id')
            .eq('set_num', set_num)
            .eq('booklet_number', booklet_number)
            .single()
        );
        
        const manualResults = await Promise.all(manualPromises);
        
        // Create a lookup map for manual IDs
        const manualIdMap = {};
        
        manualResults.forEach((result, index) => {
          if (result.error) {
            console.error("Error fetching manual ID:", result.error);
            return;
          }
          
          if (result.data) {
            const { set_num, booklet_number } = setBookletPairs[index];
            const key = `${set_num}|${booklet_number}`;
            manualIdMap[key] = result.data.manual_id;
          }
        });
        
        console.log("Manual ID map:", manualIdMap);
        
        // Get unique set numbers from steps
        const setNums = [...new Set(stepsData.map(step => step.set_num))];
        
        // Fetch set details for each set number
        const { data: setsData, error: setsError } = await supabase
          .from('sets')
          .select('*')
          .in('set_num', setNums);
          
        if (setsError) {
          console.error("Error fetching sets:", setsError);
          alert(`Error fetching set details: ${setsError.message}`);
          return;
        }
        
        console.log("Sets data:", setsData);
        
        // Combine steps with set information and manual IDs
        const enrichedStepsData = stepsData.map(step => {
          const setInfo = setsData.find(set => set.set_num === step.set_num) || {};
          const manualKey = `${step.set_num}|${step.booklet_number}`;
          const manual_id = manualIdMap[manualKey];
          
          return {
            ...step,
            set_name: setInfo.name || '',
            set_year: setInfo.year || '',
            set_img_url: setInfo.img_url || '',
            manual_id
          };
        });
        
        setSearchResults(enrichedStepsData);
      } else {
        // Try a partial match
        const { data: partialMatch, error: partialError } = await supabase
          .from('elements')
          .select('*')
          .ilike('element_id', `%${elementId.trim()}%`);
          
        if (partialError) {
          console.error("Error in partial search:", partialError);
          return;
        }
        
        console.log("Partial match results:", partialMatch);
        
        if (!partialMatch || partialMatch.length === 0) {
          alert(`No elements found matching "${elementId}"`);
        } else if (partialMatch.length === 1) {
          // Found one partial match, use it
          alert(`Found element with ID: ${partialMatch[0].element_id}`);
          setElementId(partialMatch[0].element_id);
          handleSearch(); // Recursively search with the exact ID
        } else {
          // Found multiple matches
          alert(`Found ${partialMatch.length} matching elements. Please be more specific.`);
          console.log("Multiple matches:", partialMatch);
        }
      }
    } catch (error) {
      console.error('Error during search:', error);
      alert(`Error during search: ${error.message}`);
    }
  };

  const groupBySetNumber = (results) =>
    results.reduce((acc, record) => {
      const { set_num: setNumber } = record;
      if (!acc[setNumber]) acc[setNumber] = [];
      acc[setNumber].push(record);
      return acc;
    }, {});

  const groupedResults = groupBySetNumber(searchResults);

  console.log('Grouped Results:', groupedResults); // Debugging: Log grouped results

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6 text-center">BrickMapper</h1>

      {/* Search Section */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Search Steps by Element ID</h2>
        <div className="flex items-center gap-4 mb-4">
          <input
            type="text"
            value={elementId}
            onChange={(e) => setElementId(e.target.value)}
            placeholder="Enter Element ID"
            className="border border-gray-300 rounded px-4 py-2 w-full"
          />
          <button
            onClick={handleSearch}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Search
          </button>
        </div>
        {/* Add this near your search section */}
        <button
          onClick={runDiagnostics}
          className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600 ml-2"
        >
          Run Diagnostics
        </button>
        <button
          onClick={checkSupabaseConfig}
          className="bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600 ml-2"
        >
          Check Supabase Config
        </button>
        {elementDetails && (
          <div className="p-4 border border-gray-300 rounded mb-4">
            <h3 className="text-xl font-semibold mb-2">Element Details</h3>
            <p>
              <strong>Part Number:</strong> {elementDetails.part_num}
            </p>
            <p>
              <strong>Name:</strong> {elementDetails.part_name}
            </p>
            <p>
              <strong>Color:</strong> {elementDetails.color_name}
            </p>
            <img
              src={elementDetails.img_url}
              alt={elementDetails.part_name}
              className="w-24 mt-2"
            />
          </div>
        )}
      </section>

      {/* Search Results */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Search Results</h2>
        {Object.keys(groupedResults).map((setNumber) => {
          const records = groupedResults[setNumber];
          const { set_num, set_name, set_year, set_img_url } = records[0];
          // Extract the set_num (before the "-")
          const extractedSetNum = set_num.split('-')[0];
          return (
            <div key={setNumber} className="p-4 border border-gray-300 rounded mb-4">
              <h3 className="text-xl font-semibold mb-2">
                Set Number: {extractedSetNum}{' '}
                <a
                  href={`${BUILDING_INSTRUCTIONS_URL}/${extractedSetNum}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-500 underline"
                >
                  (Building instructions)
                </a>
              </h3>
              <p>
                <strong>Name:</strong> {set_name}
              </p>
              <p>
                <strong>Year:</strong> {set_year}
              </p>
              <img src={set_img_url} alt={set_name} className="w-64 mt-2" />
              <h4 className="text-lg font-semibold mt-4">Steps:</h4>
              {/* Container with fixed width but still scrollable */}
              <div className="max-w-md overflow-auto max-h-48 border border-gray-300 rounded">
                <table className="table-auto w-full border-collapse">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border border-gray-300 px-4 py-2">Booklet</th>
                      <th className="border border-gray-300 px-4 py-2">Page</th>
                      <th className="border border-gray-300 px-4 py-2">Step</th>
                    </tr>
                  </thead>
                  <tbody>
                    {records.map((record, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="border border-gray-300 px-4 py-2 text-center">{record.booklet_number}</td>
                        <td className="border border-gray-300 px-4 py-2 text-center">
                          {record.manual_id ? (
                            <a
                              href={`${BUILDING_INSTRUCTIONS_PDFS_URL}/${record.manual_id}.pdf#page=${record.page_number}`}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-500 underline"
                            >
                              {record.page_number}
                            </a>
                          ) : (
                            record.page_number
                          )}
                        </td>
                        <td className="border border-gray-300 px-4 py-2 text-center">{record.step_number}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          );
        })}
      </section>

      {/* CSV Upload */}
      <section>
        <h2 className="text-2xl font-semibold mb-4">Upload CSV</h2>
        <div className="flex items-center gap-4 flex-wrap">
          <input
            type="file"
            accept=".csv"
            onChange={handleCsvUpload}
            className="border border-gray-300 rounded px-4 py-2"
          />
          <button
            onClick={() => uploadCsvData('steps')}
            className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
          >
            Upload to Steps
          </button>
          <button
            onClick={() => uploadCsvData('elements')}
            className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
          >
            Upload to Elements
          </button>
          <button
            onClick={() => uploadCsvData('sets')}
            className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
          >
            Upload to Sets
          </button>
          <button
            onClick={() => uploadCsvData('manuals')}
            className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
          >
            Upload to Manuals
          </button>
        </div>
      </section>
    </div>
  );
};

export default App;