import json
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views import View
from .models import LocationData, ConstellationQuery
import google.generativeai as genai
from datetime import datetime
import math
import re
from dotenv import load_dotenv
load_dotenv()
import os

class ConstellationFinderView(View):
    def get(self, request):
        return render(request, 'Locator/index.html')

@csrf_exempt
def save_location(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            lat = data.get('latitude')
            lng = data.get('longitude')
            accuracy = data.get('accuracy', 0)
            
            if not lat or not lng:
                return JsonResponse({
                    'status': 'error', 
                    'message': 'Latitude and longitude are required'
                })
            
            location = LocationData.objects.create(
                user=request.user if request.user.is_authenticated else None,
                latitude=lat,
                longitude=lng,
                accuracy=accuracy
            )
            
            return JsonResponse({
                'status': 'success',
                'location_id': location.id,
                'message': 'Location saved successfully'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error', 
                'message': 'Invalid JSON data'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error', 
                'message': f'Error saving location: {str(e)}'
            })
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@csrf_exempt
def find_constellations(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            location_id = data.get('location_id')
            
            if not location_id:
                return JsonResponse({
                    'status': 'error', 
                    'message': 'Location ID is required'
                })
            
            # Get location from database
            try:
                location = LocationData.objects.get(id=location_id)
            except LocationData.DoesNotExist:
                return JsonResponse({
                    'status': 'error', 
                    'message': 'Location not found'
                })
            
            # Configure Gemini AI
            gemini_api_key = os.getenv('API-KEY')
            if not gemini_api_key:
                return JsonResponse({
                    'status': 'error', 
                    'message': 'Gemini API key not configured'
                })
            
            genai.configure(api_key="AIzaSyA_xoVq9OqFRrwX7lhHLMOM2lodCQm08Os")
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            current_time = datetime.now()
            current_hour = current_time.hour
            prompt = f"""
            Location: Latitude {location.latitude}, Longitude {location.longitude}
            Date: {current_time.strftime('%Y-%m-%d')}
            Time: {current_time.strftime('%H:%M')} ({current_hour}:00)
            
            Please provide a simple response about constellations visible from this location tonight:
            
            1. Give a brief description (2-3 sentences) of what constellations are visible
            2. List 5-7 major constellations that can be seen
            3. For each constellation, specify the general compass direction (North, South, East, West, Northeast, Northwest, Southeast, Southwest, or Overhead)
            Also tell where am i based on location.
            Keep the response simple and practical for someone using a phone compass to find constellations.
            Format: Just plain text, no special formatting or symbols.
            """
            
            # Get response from Gemini
            try:
                response = model.generate_content(prompt)
                gemini_response = response.text.strip()
                
                # Extract constellation information
                visible_constellations = extract_constellation_names(gemini_response)
                compass_directions = extract_compass_directions(gemini_response, visible_constellations)
                
                # Save query to database
                constellation_query = ConstellationQuery.objects.create(
                    location=location,
                    query_text=prompt,
                    gemini_response=gemini_response,
                    visible_constellations=visible_constellations
                )
                
                return JsonResponse({
                    'status': 'success',
                    'response': gemini_response,
                    'visible_constellations': visible_constellations,
                    'compass_directions': compass_directions,
                    'query_id': constellation_query.id
                })
                
            except Exception as e:
                return JsonResponse({
                    'status': 'error', 
                    'message': f'Gemini API error: {str(e)}'
                })
                
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error', 
                'message': 'Invalid JSON data'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error', 
                'message': f'Server error: {str(e)}'
            })
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

def extract_constellation_names(text):
    """Extract constellation names from Gemini response using improved pattern matching"""
    
    # Common constellation names to look for
    common_constellations = [
        'Ursa Major', 'Big Dipper', 'Great Bear',
        'Ursa Minor', 'Little Dipper', 'Little Bear',
        'Cassiopeia', 'The Queen',
        'Orion', 'The Hunter',
        'Leo', 'The Lion',
        'Virgo', 'The Maiden',
        'Libra', 'The Scales',
        'Scorpius', 'Scorpio', 'The Scorpion',
        'Sagittarius', 'The Archer',
        'Capricornus', 'Capricorn', 'The Sea Goat',
        'Aquarius', 'The Water Bearer',
        'Pisces', 'The Fish',
        'Aries', 'The Ram',
        'Taurus', 'The Bull',
        'Gemini', 'The Twins',
        'Cancer', 'The Crab',
        'Andromeda', 'The Princess',
        'Perseus', 'The Hero',
        'Auriga', 'The Charioteer',
        'Bootes', 'Boötes', 'The Herdsman',
        'Corona Borealis', 'Northern Crown',
        'Cygnus', 'The Swan',
        'Lyra', 'The Harp',
        'Aquila', 'The Eagle',
        'Draco', 'The Dragon',
        'Hercules', 'The Strongman',
        'Ophiuchus', 'The Serpent Bearer',
        'Serpens', 'The Serpent',
        'Canis Major', 'The Great Dog',
        'Canis Minor', 'The Little Dog',
        'Pegasus', 'The Winged Horse',
        'Cepheus', 'The King',
        'Lacerta', 'The Lizard',
        'Vela', 'The Sails',
        'Centaurus', 'The Centaur',
        'Crux', 'Southern Cross',
        'Polaris', 'North Star', 'Pole Star'
    ]
    
    found_constellations = []
    text_upper = text.upper()
    
    for constellation in common_constellations:
        # Check if constellation name appears in text
        if constellation.upper() in text_upper:
            # Use the most common name
            if constellation in ['Big Dipper', 'Great Bear']:
                if 'Ursa Major' not in found_constellations:
                    found_constellations.append('Ursa Major')
            elif constellation in ['Little Dipper', 'Little Bear']:
                if 'Ursa Minor' not in found_constellations:
                    found_constellations.append('Ursa Minor')
            elif constellation in ['The Queen']:
                if 'Cassiopeia' not in found_constellations:
                    found_constellations.append('Cassiopeia')
            elif constellation in ['The Hunter']:
                if 'Orion' not in found_constellations:
                    found_constellations.append('Orion')
            elif constellation in ['Scorpio', 'The Scorpion']:
                if 'Scorpius' not in found_constellations:
                    found_constellations.append('Scorpius')
            elif constellation in ['Capricorn', 'The Sea Goat']:
                if 'Capricornus' not in found_constellations:
                    found_constellations.append('Capricornus')
            elif constellation in ['Boötes', 'The Herdsman']:
                if 'Bootes' not in found_constellations:
                    found_constellations.append('Bootes')
            elif constellation in ['North Star', 'Pole Star']:
                if 'Polaris' not in found_constellations:
                    found_constellations.append('Polaris')
            elif constellation in ['Southern Cross']:
                if 'Crux' not in found_constellations:
                    found_constellations.append('Crux')
            else:
                # Use the constellation name as-is if it's not an alias
                clean_name = constellation.replace('The ', '')
                if clean_name not in found_constellations:
                    found_constellations.append(clean_name)
    
    # Remove duplicates while preserving order
    unique_constellations = []
    for constellation in found_constellations:
        if constellation not in unique_constellations:
            unique_constellations.append(constellation)
    
    return unique_constellations[:8]  # Limit to 8 constellations

def extract_compass_directions(text, constellations):
    """Extract compass directions for each constellation from Gemini response"""
    
    directions = {}
    
    # Direction keywords mapping
    direction_keywords = {
        'north': ['North', 'N'],
        'northeast': ['Northeast', 'NE'],
        'east': ['East', 'E'],
        'southeast': ['Southeast', 'SE'],
        'south': ['South', 'S'],
        'southwest': ['Southwest', 'SW'],
        'west': ['West', 'W'],
        'northwest': ['Northwest', 'NW'],
        'overhead': ['Overhead', 'Zenith', 'Above']
    }
    
    # Default directions for common constellations (backup)
    default_directions = {
        'Ursa Major': 'North',
        'Ursa Minor': 'North',
        'Cassiopeia': 'North',
        'Polaris': 'North',
        'Orion': 'South',
        'Leo': 'South',
        'Scorpius': 'South',
        'Sagittarius': 'South',
        'Cygnus': 'Overhead',
        'Lyra': 'West',
        'Aquila': 'East',
        'Andromeda': 'Northeast',
        'Perseus': 'Northeast',
        'Pegasus': 'West',
        'Taurus': 'West',
        'Gemini': 'East'
    }
    
    # Try to extract directions from text
    for constellation in constellations:
        found_direction = None
        
        # Look for constellation name followed by direction
        for direction_key, direction_names in direction_keywords.items():
            for direction_name in direction_names:
                # Pattern: constellation name followed by direction within reasonable distance
                pattern = rf'{re.escape(constellation)}.*?{re.escape(direction_name.lower())}'
                if re.search(pattern, text.lower(), re.IGNORECASE):
                    found_direction = direction_names[0]  # Use first (full) name
                    break
                
                # Pattern: direction followed by constellation name
                pattern = rf'{re.escape(direction_name.lower())}.*?{re.escape(constellation)}'
                if re.search(pattern, text.lower(), re.IGNORECASE):
                    found_direction = direction_names[0]
                    break
                    
            if found_direction:
                break
        
        # Use found direction or fall back to default
        if found_direction:
            directions[constellation] = found_direction
        elif constellation in default_directions:
            directions[constellation] = default_directions[constellation]
        else:
            # Random fallback direction
            import random
            directions[constellation] = random.choice(['North', 'South', 'East', 'West', 'Northeast', 'Northwest', 'Southeast', 'Southwest'])
    
    return directions