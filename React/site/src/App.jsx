import React, { useEffect, useState } from 'react';
import { createClient } from '@supabase/supabase-js';
import './App.css'; // Import the CSS file

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseKey = import.meta.env.VITE_SUPABASE_KEY;

const supabase = createClient(supabaseUrl, supabaseKey);
const BUILDING_INSTRUCTIONS_URL = "https://www.lego.com/en-gb/service/building-instructions";
const BUILDING_INSTRUCTIONS_PDFS_URL = "https://www.lego.com/cdn/product-assets/product.bi.core.pdf";
const BRICKOGNIZE_API_URL = "https://api.brickognize.com/predict/parts/";

const App = () => {
  const [elementId, setElementId] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [elementDetails, setElementDetails] = useState(null);
  const [searchMode, setSearchMode] = useState('id'); // 'id' or 'image'
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [brickognizeResults, setBrickognizeResults] = useState([]);
  const [showPartSelection, setShowPartSelection] = useState(false);
  const [showElementSelection, setShowElementSelection] = useState(false);
  const [elementVariants, setElementVariants] = useState([]);
  const [selectedBrickognizePart, setSelectedBrickognizePart] = useState(null);
  const [selectedElementId, setSelectedElementId] = useState(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedImage(file);
      // Reset previous results
      setBrickognizeResults([]);
      setShowPartSelection(false);
      setShowElementSelection(false);
      setElementVariants([]);
      setSelectedElementId(null);
      setSearchResults([]);
      setElementDetails(null);
    }
  };

  const analyzeImage = async () => {
    if (!uploadedImage) {
      alert('Please upload an image first.');
      return;
    }

    setIsAnalyzing(true);
    setBrickognizeResults([]);
    setShowPartSelection(false);
    setShowElementSelection(false);

    try {
      const formData = new FormData();
      formData.append('query_image', uploadedImage);

      const response = await fetch(BRICKOGNIZE_API_URL, {
        method: 'POST',
        headers: {
          'accept': 'application/json'
        },
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Brickognize API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('Brickognize API response:', result);

      if (result.items && result.items.length > 0) {
        setBrickognizeResults(result.items);
        setShowPartSelection(true);
      } else {
        alert('No LEGO parts were recognized in this image. Please try a different image.');
      }
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert(`Error analyzing image: ${error.message}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const selectBrickognizePart = async (selectedPart) => {
    console.log('Selected part from Brickognize:', selectedPart);
    setSelectedBrickognizePart(selectedPart);
    
    try {
      // Search for elements with the matching part_num
      const { data: matchingElements, error } = await supabase
        .from('elements')
        .select('*')
        .eq('part_num', selectedPart.id);

      if (error) {
        console.error('Error searching for elements:', error);
        alert(`Error searching for elements: ${error.message}`);
        return;
      }

      console.log('Matching elements found:', matchingElements);

      if (!matchingElements || matchingElements.length === 0) {
        alert(`Part ${selectedPart.id} (${selectedPart.name}) was recognized but is not in our database.`);
        return;
      }

      if (matchingElements.length === 1) {
        // Only one element found, use it directly
        setElementId(matchingElements[0].element_id);
        setElementDetails(matchingElements[0]);
        setSelectedElementId(matchingElements[0].element_id);
        setShowPartSelection(false);
        
        // Automatically search for steps with this element
        await searchStepsForElement(matchingElements[0].element_id);
      } else {
        // Multiple elements found, show selection interface
        setElementVariants(matchingElements);
        setShowPartSelection(false);
        setShowElementSelection(true);
      }
    } catch (error) {
      console.error('Error processing selected part:', error);
      alert(`Error processing selected part: ${error.message}`);
    }
  };

  const selectElementVariant = async (selectedElement) => {
    console.log('Selected element variant:', selectedElement);
    
    setElementId(selectedElement.element_id);
    setElementDetails(selectedElement);
    setSelectedElementId(selectedElement.element_id);
    
    // Don't hide the element selection - keep it visible
    // setShowElementSelection(false);
    // setElementVariants([]);
    
    // Automatically search for steps with this element
    await searchStepsForElement(selectedElement.element_id);
  };

  const searchStepsForElement = async (elementIdToSearch) => {
    try {
      console.log("Searching for element ID:", elementIdToSearch);
      
      // Search for steps
      const { data: stepsData, error: stepsError } = await supabase
        .from('steps')
        .select('*')
        .eq('element_id', elementIdToSearch);
      
      if (stepsError) {
        console.error("Error searching steps:", stepsError);
        alert(`Error searching for steps: ${stepsError.message}`);
        return;
      }
      
      console.log("Steps data:", stepsData);
      
      if (!stepsData || stepsData.length === 0) {
        alert(`Element found but no steps contain this element.`);
        setSearchResults([]);
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
    } catch (error) {
      console.error('Error during search:', error);
      alert(`Error during search: ${error.message}`);
    }
  };

  const handleSearch = async () => {
    if (searchMode === 'image') {
      await analyzeImage();
      return;
    }

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
        await searchStepsForElement(exactMatch[0].element_id);
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
          setElementDetails(partialMatch[0]);
          await searchStepsForElement(partialMatch[0].element_id);
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
    <div className="container mx-auto p-6 flex flex-col min-h-screen">
      {/* Header Image */}
      <div className="text-center mb-6">
        <img 
          src="/header.png" 
          alt="BrickMapper" 
          className="mx-auto max-w-md max-h-32"
        />
      </div>

      {/* Search Section */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4">Search Steps by Element</h2>
        
        {/* Search Mode Toggle */}
        <div className="mb-4">
          <div className="flex gap-4">
            <label className="flex items-center">
              <input
                type="radio"
                value="id"
                checked={searchMode === 'id'}
                onChange={(e) => setSearchMode(e.target.value)}
                className="mr-2"
              />
              Search by Element ID
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                value="image"
                checked={searchMode === 'image'}
                onChange={(e) => setSearchMode(e.target.value)}
                className="mr-2"
              />
              Search by Image
            </label>
          </div>
        </div>

        {/* Search Input */}
        {searchMode === 'id' ? (
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
        ) : (
          <div className="mb-4">
            <div className="flex items-center gap-4 mb-4">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="border border-gray-300 rounded px-4 py-2"
              />
              <button
                onClick={handleSearch}
                disabled={!uploadedImage || isAnalyzing}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze Image'}
              </button>
            </div>
            
            {/* Brickognize Results Selection */}
            {showPartSelection && brickognizeResults.length > 0 && (
              <div className="border border-gray-300 rounded p-4 mb-4">
                <h3 className="text-lg font-semibold mb-3">Select the correct part:</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {brickognizeResults.slice(0, 6).map((item, index) => (
                    <div
                      key={index}
                      onClick={() => selectBrickognizePart(item)}
                      className="border border-gray-200 rounded p-3 cursor-pointer hover:bg-gray-50 hover:border-blue-300 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <img
                          src={item.img_url}
                          alt={item.name}
                          className="w-16 h-16 object-contain"
                          onError={(e) => {
                            e.target.style.display = 'none';
                          }}
                        />
                        <div className="flex-1">
                          <p className="font-medium text-sm">{item.name}</p>
                          <p className="text-xs text-gray-600">Part ID: {item.id}</p>
                          <p className="text-xs text-gray-500">
                            Confidence: {Math.round(item.score * 100)}%
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                {brickognizeResults.length > 6 && (
                  <p className="text-sm text-gray-500 mt-2">
                    Showing top 6 results out of {brickognizeResults.length} matches
                  </p>
                )}
              </div>
            )}

            {/* Element Variant Selection */}
            {showElementSelection && elementVariants.length > 0 && (
              <div className="border border-gray-300 rounded p-4 mb-4">
                <h3 className="text-lg font-semibold mb-3">
                  Multiple color variants found for "{selectedBrickognizePart?.name}" (Part {selectedBrickognizePart?.id}). 
                  Select a color variant to search for:
                </h3>
                <div className="overflow-x-auto">
                  <div className="max-h-128 max-w-xl overflow-y-auto">
                    <table className="table-auto border-collapse min-w-max w-full">
                      <thead className="sticky top-0 bg-gray-100 z-10">
                        <tr>
                          <th className="px-4 py-2">Element ID</th>
                          <th className="px-4 py-2">Color</th>
                          <th className="px-4 py-2">Image</th>
                          <th className="px-4 py-2">Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {elementVariants.map((element, index) => (
                          <tr
                            key={index}
                            className={`hover:bg-gray-50 ${selectedElementId === element.element_id ? 'bg-blue-50 border-blue-300' : ''}`}
                          >
                            <td className="font-mono text-sm whitespace-nowrap px-4 py-2">
                              {element.element_id}
                            </td>
                            <td className="whitespace-nowrap px-4 py-2">
                              <span className="inline-block px-2 py-1 bg-gray-200 rounded text-sm">
                                {element.color_name || 'Unknown Color'}
                              </span>
                            </td>
                            <td className="whitespace-nowrap px-4 py-2">
                              {element.img_url ? (
                                <img
                                  src={element.img_url}
                                  alt={element.part_name}
                                  className="w-12 h-12 object-contain"
                                  onError={(e) => {
                                    e.target.style.display = 'none';
                                  }}
                                />
                              ) : (
                                <span className="text-gray-400 text-xs">No image</span>
                              )}
                            </td>
                            <td className="whitespace-nowrap px-4 py-2">
                              <button
                                onClick={() => selectElementVariant(element)}
                                className={`px-3 py-1 rounded text-sm transition-colors ${
                                  selectedElementId === element.element_id
                                    ? 'bg-blue-500 text-white hover:bg-blue-600'
                                    : 'bg-green-500 text-white hover:bg-green-600'
                                }`}
                              >
                                {selectedElementId === element.element_id ? 'Selected' : 'Select'}
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                <div className="mt-3 flex justify-end">
                  <button
                    onClick={() => {
                      setShowElementSelection(false);
                      setShowPartSelection(true);
                      setElementVariants([]);
                      setSelectedElementId(null);
                      setSearchResults([]);
                      setElementDetails(null);
                    }}
                    className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600 transition-colors"
                  >
                    Back to Part Selection
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
        
        {elementDetails && (
          <div className="p-4 border border-gray-300 rounded mb-4">
            <h3 className="text-xl font-semibold mb-2">Element Details</h3>
            <p>
              <strong>Element ID:</strong> {elementDetails.element_id}
            </p>
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
      {Object.keys(groupedResults).length > 0 && (
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
      )}
      {/* Footer */}
      <footer className="text-center mt-8 py-4 bg-gray-100">
        <p>
          Copyright © {new Date().getFullYear()} Idan Dekel. All rights reserved.
        </p>
        <p>
          <a href="/about" className="text-blue-500 hover:underline mr-4">
            About
          </a>
          <a href="/terms" className="text-blue-500 hover:underline">
            Terms of Service
          </a>
        </p>
      </footer>
    </div>
  );
};

export default App;