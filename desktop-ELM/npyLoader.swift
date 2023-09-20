//
//  dataLoader.swift
//  desktop-ELM
//
//  Created by Anton Akusok on 03/08/2018.
//  Copyright © 2018 Anton Akusok. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

let MAGIC_PREFIX = "\u{93}NUMPY"

public struct Npy {
    let elementsData: Data
    let shape: [Int]
    let dataType: DataType
    let endian: Endian
    let isFortranOrder: Bool
    let descr: String

    var count: Int {
        return shape.reduce(1, *)
    }
    
    public var rows: Int {
        return shape.count == 1 ? 1 : shape[0]
    }
    
    public var columns: Int {
        return shape.count == 1 ? shape[0] : shape[1]
    }
    
    public var mpsDataType: MPSDataType {
        return DataType.mpsType[dataType]!
    }
    
    public var stride: Int {
        return DataType.stride[dataType]!
    }
    
//    mutating func addRows(_ n: Int) {
//        self.elementsData.append(contentsOf: Array<Float>.init(repeating: 0.0, count: n * self.columns))
//    }
//    
    init(elementsData: Data, shape: [Int], dataType: DataType, endian: Endian, isFortranOrder: Bool, descr: String) {
        self.elementsData = elementsData
        self.shape = shape
        self.dataType = dataType
        self.endian = endian
        self.isFortranOrder = isFortranOrder
        self.descr = descr
    }
}

extension Npy {
    
    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        try self.init(data: data)
    }

    public init(data: Data) throws {
        guard let magic = String(data: data.subdata(in: 0..<6), encoding: .ascii) else {
            throw NpyLoaderError.ParseFailed(message: "Can't parse prefix")
        }
        guard magic == MAGIC_PREFIX else {
            throw NpyLoaderError.ParseFailed(message: "Invalid prefix: \(magic)")
        }
        
        let major = data[6]
        guard major == 1 || major == 2 else {
            throw NpyLoaderError.ParseFailed(message: "Invalid major version: \(major)")
        }
        
        let minor = data[7]
        guard minor == 0 else {
            throw NpyLoaderError.ParseFailed(message: "Invalid minor version: \(minor)")
        }
        
        let headerLen: Int
        let headerStarts: Int
        switch major {
        case 1:
            let tmp = data[8...9].withUnsafeBytes { UInt16(littleEndian: $0.pointee) }
            headerLen = Int(tmp)
            headerStarts = 10
        case 2:
            let tmp = data[8...11].withUnsafeBytes { UInt32(littleEndian: $0.pointee) }
            headerLen = Int(tmp)
            headerStarts = 12
        default:
            fatalError("Never happens.")
        }
        
        let headerData = data.subdata(in: headerStarts ..< headerStarts+headerLen)
        guard let str = String(data: headerData, encoding: .ascii) else {
            throw NpyLoaderError.ParseFailed(message: "Failed to load header")
        }
        
        let descr: String
        let endian: Endian
        let dataType: DataType
        let isFortranOrder: Bool
        do {
            let separate = str.components(separatedBy: CharacterSet(charactersIn: ", ")).filter { !$0.isEmpty }
            
            guard let descrIndex = separate.index(where: { $0.contains("descr") }) else {
                throw NpyLoaderError.ParseFailed(message: "Header does not contain the key 'descr'")
            }
            descr = separate[descrIndex + 1]
            
            guard let e = Endian.all.filter({ descr.contains($0.rawValue) }).first else {
                throw NpyLoaderError.ParseFailed(message: "Unknown endian")
            }
            endian = e
            
            guard let dt = DataType.all.filter({ descr.contains($0.rawValue) }).first else {
                throw NpyLoaderError.UnsupportedFormat(message: "Unsupported dtype: \(descr)")
            }
            dataType = dt
            
            guard let fortranIndex = separate.index(where: { $0.contains("fortran_order") }) else {
                throw NpyLoaderError.ParseFailed(message: "Header does not contain the key 'fortran_order'")
            }
            
            isFortranOrder = separate[fortranIndex+1].contains("True")
            guard !isFortranOrder else {
                throw NpyLoaderError.UnsupportedFormat(message: "Fortran-ordered data is unsupported.")
            }
        }
        
        var shape: [Int] = []
        do {
            guard let left = str.range(of: "("),
                let right = str.range(of: ")") else {
                    throw NpyLoaderError.ParseFailed(message: "Shape not found in header.")
            }
            
            let substr = str[left.upperBound..<right.lowerBound]
            
            let strs = substr.replacingOccurrences(of: " ", with: "")
                .components(separatedBy: ",")
                .filter { !$0.isEmpty }
            for s in strs {
                guard let i = Int(s) else {
                    throw NpyLoaderError.ParseFailed(message: "Shape contains invalid integer: \(s)")
                }
                shape.append(i)
            }
        }
        guard shape.count == 2 || shape.count == 1 else {
            throw NpyLoaderError.UnsupportedFormat(message: "Only (1 or 2)-dimensional matrices are supported.")
        }
        
//        print("Loading \(shape) matrix of \(dataType) elements, \(endian) endian and in \(isFortranOrder ? "Fortran" : "C")-ordering; description \(descr).")
        
        let rows = shape.count == 1 ? 1 : shape[0]
        let columns = shape.count == 1 ? shape[0] : shape[1]
        let rowBytes = columns * DataType.stride[dataType]!
        let content = data.suffix(from: headerStarts+headerLen)
        
        guard content.count == rows * rowBytes else {
            throw NpyLoaderError.ParseFailed(message: "Incorrect data size: get \(content.count) data bytes for \(rows) rows and \(columns) columns of type \(dataType).")
        }
        
        self.init(elementsData: content, shape: shape, dataType: dataType, endian: endian, isFortranOrder: isFortranOrder, descr: descr)
    }
}

public enum NpyLoaderError: Error {
    case ParseFailed(message: String)
    case UnsupportedFormat(message: String)
}

public enum Endian: String {
    case host = "="
    case big = ">"
    case little = "<"
    case na = "|"
    
    static var all: [Endian] {
        return [.host, .big, .little, .na]
    }
}

public enum DataType: String {
    case uint8 = "u1"
    case uint16 = "u2"
    case uint32 = "u4"
    
    case int8 = "i1"
    case int16 = "i2"
    
    case float16 = "f2"
    case float32 = "f4"
    
    static var all: [DataType] {
        return [.uint8, .uint16, .uint32, .int8, .int16, .float16, .float32]
    }
    
    static let mpsType: [DataType: MPSDataType] = [.int8: .int8, .int16: .int16,
                                                   .uint8: .uInt8, .uint16:.uInt16, .uint32:.uInt32,
                                                   .float16:.float16, .float32:.float32]
    
    static let stride: [DataType: Int] = [.int8: MemoryLayout<Int8>.stride,
                                          .int16: MemoryLayout<Int16>.stride,
                                          .uint8: MemoryLayout<UInt8>.stride,
                                          .uint16: MemoryLayout<UInt16>.stride,
                                          .uint32: MemoryLayout<UInt32>.stride,
                                          .float16: MemoryLayout<UInt16>.stride,
                                          .float32: MemoryLayout<Float32>.stride]
}


///// Loads Numpy data file to MPS Matrix.
///// Preserves data type.
/////
///// - parameter contentsOf: URL of the Numpy file
///// - parameter mode: storage mode, default is shared
///// - returns: MPSMatrix with correct shape and data type
//public func loadNpyToMatrix(contentsOf url: URL, mode: MTLResourceOptions = .storageModeShared) -> MPSMatrix {
//    let npy = try! Npy(contentsOf: url)
//    let rowBytes = npy.columns * DataType.stride[npy.dataType]!
//    let matDescriptor = MPSMatrixDescriptor(rows: npy.rows, columns: npy.columns, rowBytes: rowBytes, dataType: npy.mpsDataType)
//    let matBuffer = npy.elementsData.withUnsafeBytes { device.makeBuffer(bytes: $0, length: npy.count * npy.stride, options: [mode])! }
//    return MPSMatrix(buffer: matBuffer, descriptor: matDescriptor)
//}
//
//
///// Loads Numpy data file to MPS Vector.
///// Preserves data type.
/////
///// - parameter contentsOf: URL of the Numpy file
///// - parameter mode: storage mode, default is shared
///// - returns: MPSVector with correct shape and data type
//public func loadNpyToVector(contentsOf url: URL, mode: MTLResourceOptions = .storageModeShared) -> MPSVector {
//    let npy = try! Npy(contentsOf: url)
//    let vectDescriptor = MPSVectorDescriptor(length: npy.columns, dataType: npy.mpsDataType)
//    let vectBuffer = npy.elementsData.withUnsafeBytes { device.makeBuffer(bytes: $0, length: npy.count * npy.stride, options: [mode])! }
//    return MPSVector(buffer: vectBuffer, descriptor: vectDescriptor)
//}

