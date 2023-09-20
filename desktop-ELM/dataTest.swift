//
//  dataTest.swift
//  desktop-ELM
//
//  Created by Anton Akusok on 02/08/2018.
//  Copyright © 2018 Anton Akusok. All rights reserved.
//

import Foundation


//func npyFoo<T>(fileX: URL) throws -> UnsafeBufferPointer<T> {
func npyFoo(fileX: URL) throws {

    let data = NSData(contentsOf: fileX)!
        
    let magic = String(data: data.subdata(with: NSRange(location: 0, length: 6)), encoding: .ascii)!
    print(magic, magic == MAGIC_PREFIX)

    let headerLen: Int
    let headerStarts: Int

    let major = data.subdata(with: NSRange(location: 6, length: 1))[0]
    switch major {
    case 1:
        var tmp : UInt16 = 0
        data.getBytes(&tmp, range: NSRange(location: 8, length: 2))
        headerLen = Int(tmp)
        headerStarts = 10
    case 2:
        var tmp : UInt32 = 0
        data.getBytes(&tmp, range: NSRange(location: 8, length: 4))
        headerLen = Int(tmp)
        headerStarts = 12
    default:
        throw NpyLoaderError.ParseFailed(message: "Invalid major version: \(major)")
    }
    
    
    let str = String.init(data: data.subdata(with: NSMakeRange(headerStarts, headerLen)), encoding: .ascii)!
//    let str = "{'descr': '<f4', 'fortran_order': False, 'shape': (16384, 16384), }"
    let separate = str.components(separatedBy: CharacterSet(charactersIn: ", ")).filter { !$0.isEmpty }

//    let descrIndex = separate.index(where: { $0.contains("descr") })!
//    let descr = separate[descrIndex + 1]
//    let dt = DataType.all.filter({ descr.contains($0.rawValue) }).first!
//    let dataType = dt

    let fortranIndex = separate.index(where: { $0.contains("fortran_order") })!
    if separate[fortranIndex+1].contains("True") {
        throw NpyLoaderError.TypeMismatch(message: "Fortran ordering not supported!")
    }
    print(separate)

    let substr = str[str.range(of: "(")!.upperBound ..< str.range(of: ")")!.lowerBound]
    let shape = substr.replacingOccurrences(of: " ", with: "")
        .components(separatedBy: ",")
        .filter { !$0.isEmpty }
        .map { Int($0)! }
    print(shape)
    
//    let n = shape.reduce(1, *) * MemoryLayout<T>.stride
//    let ptrX = UnsafeMutableRawPointer.allocate(byteCount: n, alignment: 0)
//    data.getBytes(ptrX, range: NSRange(location: headerStarts + headerLen, length: n))
//
//    let X = UnsafeBufferPointer(start: ptrX.assumingMemoryBound(to: T.self), count: shape.reduce(1, *))
//
    print("lol wutz")
//    return X
}
