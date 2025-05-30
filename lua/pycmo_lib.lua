function move_unit_to(side, unit_name, latitude, longitude)
    if latitude >= -90 and latitude <= 90 and longitude >= -180 and longitude <= 180 then
        ScenEdit_SetUnit({side = side, unitname = unit_name, course = {{longitude = longitude, latitude = latitude, TypeOf = 'ManualPlottedCourseWaypoint'}}})
    end
end

function ScenarioHasEnded(ended)
    local scenario = VP_GetScenario()
    WriteData(tostring(ended), scenario.Title .. '_scen_has_ended.inst')
end

-- Functions to emulate ScenEdit_ExportScenarioToXML()
function ScenEdit_ExportScenarioToXML()
    local scenario_xml = "<?xml version='1.0' encoding='utf-8'?><Scenario>"

    local scenario = VP_GetScenario()
    scenario_xml = scenario_xml .. WrapInXML(scenario.Title, 'Title')
    scenario_xml = scenario_xml .. WrapInXML(scenario.FileName, 'FileName')
    scenario_xml = scenario_xml .. WrapInXML(scenario.CurrentTimeNum, 'Time')
    scenario_xml = scenario_xml .. WrapInXML(scenario.StartTimeNum, 'StartTime')
    scenario_xml = scenario_xml .. WrapInXML(scenario.StartTimeNum, 'ZeroHour')
    scenario_xml = scenario_xml .. WrapInXML(scenario.DurationNum, 'Duration')
    scenario_xml = scenario_xml .. WrapInXML(scenario.SaveVersion, 'SaveVersion')
    scenario_xml = scenario_xml .. WrapInXML(scenario.CampaignScore, 'CampaignScore')
    scenario_xml = scenario_xml .. WrapInXML(scenario.HasStarted, 'HasStarted')
    scenario_xml = scenario_xml .. WrapInXML(scenario.Status, 'Status')
    scenario_xml = scenario_xml .. WrapInXML(scenario.TimeCompression, 'TimeCompression')
    scenario_xml = scenario_xml .. WrapInXML(scenario.GameStatus, 'GameStatus')
    scenario_xml = scenario_xml .. WrapInXML(ExportSidesToXML(), 'Sides')
    scenario_xml = scenario_xml .. WrapInXML(ExportUnitsToXML(), 'ActiveUnits')

    scenario_xml = scenario_xml .. '</Scenario>'

    WriteData(scenario_xml, scenario.Title .. '.inst')
    -- WriteData(scenario_xml, scenario.Title .. '.xml')
end

function ExportSidesToXML()
    local sides_xml = ""

    local sides = VP_GetSides()

    for side_idx = 1, #sides do
        local side = sides[side_idx]

        sides_xml = sides_xml .. "<Side>"

        sides_xml = sides_xml .. WrapInXML(side.guid, 'ID')
        sides_xml = sides_xml .. WrapInXML(side.name, 'Name')
        sides_xml = sides_xml .. WrapInXML(ScenEdit_GetScore(side.name), 'TotalScore')
        -- sides_xml = sides_xml .. WrapInXML('', 'Missions')
        sides_xml = sides_xml .. WrapInXML(ExportMissionsToXML(side.name), 'Missions')
        sides_xml = sides_xml .. WrapInXML('', 'Prof')
        sides_xml = sides_xml .. WrapInXML('', 'Doctrine')
        sides_xml = sides_xml .. ExportContactsToXML(side.name)

        sides_xml = sides_xml .. "</Side>"
    end

    return sides_xml
end

function ExportContactsToXML(side_name)
    local contacts_xml = ""

    local contacts = ScenEdit_GetContacts(side_name)

    for contact_idx = 1, #contacts do
        local contact = contacts[contact_idx]
        contacts_xml = contacts_xml .. ExportContactToXML(contact)
    end
    
    return WrapInXML(contacts_xml, "Contacts")
end

function ExportContactToXML(contact)
    local contact_xml = ""

    contact_xml = contact_xml .. WrapInXML(contact.guid, "ID")
    contact_xml = contact_xml .. WrapInXML(contact.name, "Name")
    contact_xml = contact_xml .. WrapInXML(contact.type, "Type")
    if contact.altitude ~= nil then contact_xml = contact_xml .. WrapInXML(contact.altitude, "CA") end
    if contact.speed ~= nil then contact_xml = contact_xml .. WrapInXML(contact.speed, "CS") end
    if contact.latitude ~= nil then contact_xml = contact_xml .. WrapInXML(contact.latitude, "Lat") end
    if contact.longitude ~= nil then contact_xml = contact_xml .. WrapInXML(contact.longitude, "Lon") end
    
    return WrapInXML(contact_xml, "Contact")
end

function ExportUnitsToXML()
    local units_xml = ""

    local sides = VP_GetSides()
    for side_idx = 1, #sides do
        local side = sides[side_idx]
        local side_units = side.units
        for side_unit_idx = 1, #side_units do
            local side_unit = side_units[side_unit_idx]
            units_xml = units_xml .. ExportUnitToXML(side_unit.guid)
        end
    end

    return units_xml
end

function ExportUnitToXML(guid)
    local unit_xml = ""
    local unit = ScenEdit_GetUnit({guid = guid})

    if unit == nil then
        return ''
    end

    if unit.type == 'Facility' then -- there is a limit to the length of the comment that we can export
        return ''
    end


    unit_xml = unit_xml .. WrapInXML(unit.guid, 'ID')
    unit_xml = unit_xml .. WrapInXML(unit.dbid, 'DBID')
    unit_xml = unit_xml .. WrapInXML(unit.name, 'Name')
    unit_xml = unit_xml .. WrapInXML(unit.side, 'Side')
    unit_xml = unit_xml .. WrapInXML(unit.classname, 'ClassName')
    unit_xml = unit_xml .. WrapInXML(unit.proficiency, 'Proficiency')
    unit_xml = unit_xml .. WrapInXML(unit.latitude, 'Lat')
    unit_xml = unit_xml .. WrapInXML(unit.longitude, 'Lon')
    unit_xml = unit_xml .. WrapInXML(unit.altitude, 'CA')
    unit_xml = unit_xml .. WrapInXML(unit.heading, 'CH')
    unit_xml = unit_xml .. WrapInXML(unit.speed, 'CS')
    unit_xml = unit_xml .. WrapInXML(unit.throttle, 'Thr')
    unit_xml = unit_xml .. WrapInXML(unit.fuelstate, 'FuelState')
    unit_xml = unit_xml .. WrapInXML(unit.weaponstate, 'WeaponState')
    if unit.loadoutdbid ~= nil then
        unit_xml = unit_xml .. WrapInXML(ExportUnitLoadoutToXML(unit.name), 'Loadout')
    end
    if unit.mounts ~= nil then
        unit_xml = unit_xml .. ExportUnitMountsToXML(unit)
    end
    if unit.fuel ~= nil then
        unit_xml = unit_xml .. ExportUnitFuelsToXML(unit)
    end
    -- unit_xml = unit_xml .. WrapInXML('', 'Doctrine')
    -- unit_xml = unit_xml .. WrapInXML('', 'Sensors')
    -- unit_xml = unit_xml .. WrapInXML('', 'Comms')
    -- unit_xml = unit_xml .. WrapInXML('', 'Propulsion')
    
    return WrapInXML(unit_xml, unit.type)
end

function ExportUnitFuelsToXML(unit)
    local fuels_xml = ""

    local unit_fuels = unit.fuel

    for fuel_type, fuel in pairs(unit_fuels) do
        fuels_xml = fuels_xml .. ExportFuelToXML(fuel)
    end

    if fuels_xml ~= "" then
        return WrapInXML(fuels_xml, "Fuel")
    else
        return fuels_xml
    end
    
end

function ExportFuelToXML(fuel)
    local fuel_xml = ''

    fuel_xml = fuel_xml .. WrapInXML(fuel.type, "FT")
    fuel_xml = fuel_xml .. WrapInXML(fuel.current, "CQ")
    fuel_xml = fuel_xml .. WrapInXML(fuel.max, "MQ")

    return WrapInXML(fuel_xml, "FuelRec")
end

function ExportUnitLoadoutToXML(unitname)
    local loadout_xml = ""

    local unit_loadout = ScenEdit_GetLoadout({unitname = unitname})

    loadout_xml = loadout_xml .. WrapInXML(unit_loadout.dbid, 'ID')
    loadout_xml = loadout_xml .. WrapInXML(unit_loadout.dbid, 'DBID')
    loadout_xml = loadout_xml .. WrapInXML(unit_loadout.name, 'Name')
    loadout_xml = loadout_xml .. WrapInXML(ExportUnitLoadoutWeaponsToXML(unitname), 'Weaps')

    return WrapInXML(loadout_xml, "Loadout")
end

function ExportUnitLoadoutWeaponsToXML(unitname)
    local weapons_xml = ""

    local loadout_weapons = ScenEdit_GetLoadout({unitname = unitname}).weapons

    if #loadout_weapons == 0 then return '' end

    for weapon_idx = 1, #loadout_weapons do
        local loadout_weapon = loadout_weapons[weapon_idx]
        weapons_xml = weapons_xml .. ExportWeaponToXML(loadout_weapon)
    end

    return weapons_xml
end

function ExportUnitMountsToXML(unit)
    local mounts_xml = ""

    local unit_mounts = unit.mounts

    if #unit_mounts == 0 then return '' end

    for mount_idx = 1, #unit_mounts do
        local unit_mount = unit_mounts[mount_idx]
        mounts_xml = mounts_xml .. ExportMountToXML(unit_mount)
    end

    return WrapInXML(mounts_xml, "Mounts")
end

function ExportMountToXML(mount)
    local mount_xml = ""

    mount_xml = mount_xml .. WrapInXML(mount.mount_guid, 'ID')
    mount_xml = mount_xml .. WrapInXML(mount.mount_dbid, 'DBID')
    mount_xml = mount_xml .. WrapInXML(mount.mount_name, 'Name')
    mount_xml = mount_xml .. WrapInXML(mount.mount_status, 'MountStatus')
    if mount.mount_weapons ~= nil then mount_xml = mount_xml .. WrapInXML(ExportUnitMountWeaponsToXML(mount.mount_weapons), 'MW') end

    return WrapInXML(mount_xml, "Mount")
end

function ExportUnitMountWeaponsToXML(mount_weapons)
    local weapons_xml = ""

    if #mount_weapons == 0 then return '' end

    for weapon_idx = 1, #mount_weapons do
        local mount_weapon = mount_weapons[weapon_idx]
        weapons_xml = weapons_xml .. ExportWeaponToXML(mount_weapon)
    end

    return weapons_xml
end

function ExportWeaponToXML(weapon)
    local weapon_xml = ""
    
    weapon_xml = weapon_xml .. WrapInXML(weapon.wpn_guid, 'ID')
    weapon_xml = weapon_xml .. WrapInXML(weapon.wpn_dbid, 'WeapID')
    if weapon.wpn_current ~= nil then weapon_xml = weapon_xml .. WrapInXML(weapon.wpn_current, 'CL') end
    if weapon.wpn_maxcap ~= nil then  weapon_xml = weapon_xml .. WrapInXML(weapon.wpn_maxcap, 'ML') end
 
    return WrapInXML(weapon_xml, "WRec")
 end

function ExportMissionsToXML(side_name)
    local missions_xml = ""
    local missions = ScenEdit_GetMissions(side_name)

    for mission_idx = 1, #missions do
        local mission = missions[mission_idx]
        missions_xml = missions_xml .. ExportMissionToXML(mission)
    end

    return missions_xml
end

function ExportMissionToXML(mission)
    local mission_xml = ""

    mission_xml = mission_xml .. WrapInXML(mission.guid, 'ID')
    mission_xml = mission_xml .. WrapInXML(mission.name, 'Name')
    mission_xml = mission_xml .. WrapInXML(mission.type, 'Type')
    mission_xml = mission_xml .. WrapInXML(mission.subtype, 'Subtype')
    mission_xml = mission_xml .. WrapInXML(mission.isactive, 'IsActive')
    mission_xml = mission_xml .. WrapInXML(mission.starttime, 'StartTime')
    mission_xml = mission_xml .. WrapInXML(mission.endtime, 'EndTime')
    mission_xml = mission_xml .. WrapInXML(mission.SISH, 'SISH')
    
    -- 导出任务单位列表
    if mission.unitlist then
        local units_xml = ""
        for _, unit in pairs(mission.unitlist) do
            units_xml = units_xml .. WrapInXML(unit, 'Unit')
        end
        mission_xml = mission_xml .. WrapInXML(units_xml, 'UnitList')
    end

    return WrapInXML(mission_xml, 'Mission')
end

function WrapInXML(data, tag)
    return '<' .. tag .. '>' .. tostring(data) .. '</' .. tag .. '>'
end

function WriteData(data, filename)
    -- must have a valid side for ScenEdit_ExportInst to work
    local sides = VP_GetSides() -- use random side to export data because we do not care about what side we use
    ScenEdit_ExportInst(sides[1].name, {}, {filename = filename, comment = data})
end

function teardown_and_end_scenario(export_observation_event_name, end_scenario)
    VP_SetTimeCompression(0)
    local scenario_events = ScenEdit_GetEvents(1)
    for i = 1, #scenario_events do
        local event = scenario_events[i]
        if event.description == export_observation_event_name then
            ScenEdit_SetEvent(event.description, {mode = 'remove'})
            break
        end
    end
    ScenEdit_ExportScenarioToXML()
    ScenarioHasEnded(true)
    if end_scenario == true then
        ScenEdit_EndScenario()
    end
end