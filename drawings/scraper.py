from lxml import html
import requests
import re

def get_hero_responses(hero_name):
  page = requests.get('http://dota2.gamepedia.com/{0}_responses'.format(hero_name))
  tree = html.fromstring(page.content)
  ret = ""
  for i in range(100):
    for j in range(1, 100):
      res1 = tree.xpath('//*[@id="mw-content-text"]/ul[{0}]/li[{1}]/text()'.format(i, j))
      if len(res1) == 0:
        break
      for x in res1:
        ret += " "
        ret += x
  regex = re.compile('[^a-zA-Z ]')
  ret = str(regex.sub('', ret))
  ret = ret.split(' ')
  ret = filter(lambda x: len(x) > 0, ret)
  return ret

# turns out it was difficult to scrape hero names so fuck it lmao
hero_names = [
"Ancient_Apparition",
"Bane",
"Batrider",
"Chen",
"Crystal_Maiden",
"Dark_Seer",
"Dazzle",
"Death_Prophet",
"Disruptor",
"Enchantress",
"Enigma",
"Invoker",
"Jakiro",
"Keeper_of_the_Light",
"Leshrac",
"Lich",
"Lina",
"Lion",
"Nature%27s_Prophet",
"Necrophos",
"Ogre_Magi",
"Oracle",
"Outworld_Devourer",
"Puck",
"Pugna",
"Queen_of_Pain",
"Rubick",
"Shadow_Demon",
"Shadow_Shaman",
"Silencer",
"Skywrath_Mage",
"Storm_Spirit",
"Techies",
"Tinker",
"Visage",
"Warlock",
"Windranger",
"Winter_Wyvern",
"Witch_Doctor",
"Zeus",
"Abaddon",
"Alchemist",
"Axe",
"Beastmaster",
"Brewmaster",
"Bristleback",
"Centaur_Warrunner",
"Chaos_Knight",
"Clockwerk",
"Doom",
"Dragon_Knight",
"Earth_Spirit",
"Earthshaker",
"Elder_Titan",
"Huskar",
"Io",
"Kunkka",
"Legion_Commander",
"Lifestealer",
"Lycan",
"Magnus",
"Night_Stalker",
"Omniknight",
"Phoenix",
"Pudge",
"Sand_King",
"Slardar",
"Spirit_Breaker",
"Sven",
"Tidehunter",
"Timbersaw",
"Tiny",
"Treant_Protector",
"Tusk",
"Undying",
"Wraith_King",
"Anti-Mage",
"Arc_Warden",
"Bloodseeker",
"Bounty_Hunter",
"Broodmother",
"Clinkz",
"Drow_Ranger",
"Ember_Spirit",
"Faceless_Void",
"Gyrocopter",
"Juggernaut",
"Lone_Druid",
"Luna",
"Medusa",
"Meepo",
"Mirana",
"Morphling",
"Naga_Siren",
"Nyx_Assassin",
"Phantom_Assassin",
"Phantom_Lancer",
"Razor",
"Riki",
"Shadow_Fiend",
"Slark",
"Sniper",
"Spectre",
"Templar_Assassin",
"Terrorblade",
"Troll_Warlord",
"Ursa",
"Vengeful_Spirit",
"Venomancer",
"Viper",
"Weaver"]

def get_data_line(hero_name):
  resp = get_hero_responses(hero_name)
  ret = hero_name
  for x in resp:
    ret += " "+x
  ret += "\n"
  return ret

fd = open("data", "w")
for hero_name in hero_names:
  print "scraping response for "+hero_name
  line = get_data_line(hero_name)
  fd.write(line)
fd.close()

# resp_name = get_hero_names()

